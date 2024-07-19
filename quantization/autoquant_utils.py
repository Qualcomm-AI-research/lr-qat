# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import copy
from typing import Dict, List, Optional, Type

from torch import nn
from torch.nn import functional as F

from quantization.base_quantized_classes import (
    FP32Acts,
    QuantizedActivation,
    QuantizedModule,
)
from quantization.hijacker import QuantizationHijacker, activations_set
from quantization.quantization_manager import QuantizationManager


class QuantLinear(QuantizationHijacker, nn.Linear):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)


class QuantizedActivationWrapper(QuantizedActivation):
    """
    Wraps over a layer and quantized the activation.
    It also allow for tying the input and output quantizer which is helpful
    for layers such Average Pooling
    """

    def __init__(
        self,
        layer,
        tie_activation_quantizers=False,
        input_quantizer: Optional[QuantizationManager] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tie_activation_quantizers = tie_activation_quantizers
        if input_quantizer:
            assert isinstance(input_quantizer, QuantizationManager)
            self.activation_quantizer = input_quantizer
        self.layer = layer

    def quantize_activations_no_range_update(self, x):
        if self._quant_a:
            return self.activation_quantizer.quantizer(x)
        else:
            return x

    def forward(self, x):
        x = self.layer(x)
        if self.tie_activation_quantizers:
            # The input activation quantizer is used to quantize the activation
            # but without updating the quantization range
            return self.quantize_activations_no_range_update(x)
        else:
            return self.quantize_activations(x)

    def extra_repr(self):
        return f"tie_activation_quantizers={self.tie_activation_quantizers}"


class QuantLayerNorm(QuantizationHijacker, nn.LayerNorm):
    def run_forward(self, x, weight, bias, offsets=None):
        return F.layer_norm(
            input=x.contiguous(),
            normalized_shape=self.normalized_shape,
            weight=weight.contiguous(),
            bias=bias.contiguous(),
            eps=self.eps,
        )


class QuantEmbedding(QuantizationHijacker, nn.Embedding):
    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
        # NB: We should not (re-)quantize activations of this module, as it is a
        # lookup table (=weights), which is already quantized
        self.activation_quantizer = FP32Acts()

    def run_forward(self, x, weight, bias, offsets=None):
        return F.embedding(
            input=x.contiguous(),
            weight=weight.contiguous(),
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


module_map = {
    nn.Linear: QuantLinear,
    nn.Embedding: QuantEmbedding,
    nn.LayerNorm: QuantLayerNorm,
}


def is_next_module_bn(modules: List[nn.Module], i: int) -> bool:
    return len(modules) > i + 1 and isinstance(modules[i + 1], (nn.BatchNorm2d, nn.BatchNorm1d))


def get_act(module, i):
    """Extract activation module from series of layers."""

    # Case 1: conv + act
    if len(module) - i > 1 and isinstance(module[i + 1], tuple(activations_set)):
        return module[i + 1], i + 1

    # Case 2: conv + bn + act
    if (
        len(module) - i > 2
        and is_next_module_bn(module, i)
        and isinstance(module[i + 2], tuple(activations_set))
    ):
        return module[i + 2], i + 2

    # Case 3: conv + bn + X -> return false
    # Case 4: conv + X -> return false
    return None, None


def get_linear_args(module: nn.Linear, avoid_weight_allocation: bool = False):
    """Get keyword arguments for instantiating a new Linear module similar to an existing one.

    Args:
        module: an existing Linear module from which we want to extract the arguments.
        avoid_weight_allocation: if True, the values of the returned arguments
            will be such that no weight will be allocated when used.
            This will speed up the instantiation of the module, useful if weights
            will be overwritten in any case after the module instantiation.

    Returns: keyword arguments that can be used for instantiating a new Linear module.
    """
    args = dict(
        in_features=0 if avoid_weight_allocation else module.in_features,
        out_features=0 if avoid_weight_allocation else module.out_features,
        bias=module.bias is not None,
    )
    return args


def get_layernorm_args(module: nn.LayerNorm):
    """Get keyword arguments for instantiating a new LayerNorm module similar to an existing one.

    Args:
        module: an existing LayerNorm module from which we want to extract the arguments.

    Returns: keyword arguments that can be used for instantiating a new LayerNorm module.
    """
    args = dict(normalized_shape=module.normalized_shape, eps=module.eps)
    return args


def get_embedding_args(module: nn.Embedding, avoid_weight_allocation: bool = False):
    """Get keyword arguments for instantiating a new Embeddings module similar to an existing one.

    Args:
        module: an existing Embedding module from which we want to extract the arguments.
        avoid_weight_allocation: if True, the values of the returned arguments
            will be such that a minimal amount of weight will be allocated when used.
            This will speed up the instantiation of the module, useful if weights
            will be overwritten in any case after the module instantiation.

    Returns: keyword arguments that can be used for instantiating a new Embeddings module.
    """
    args = dict(
        num_embeddings=2 if avoid_weight_allocation else module.num_embeddings,
        embedding_dim=1 if avoid_weight_allocation else module.embedding_dim,
        padding_idx=1 if avoid_weight_allocation else module.padding_idx,
        max_norm=module.max_norm,
        norm_type=module.norm_type,
        scale_grad_by_freq=module.scale_grad_by_freq,
        sparse=module.sparse,
    )
    return args


def get_module_args(mod: nn.Module, act, avoid_weight_allocation: bool = False):
    """Get keyword arguments for instantiating a new module similar to an existing one.

    Args:
        mod: an existing Module from which we want to extract the arguments.
        act:
        avoid_weight_allocation: if True, the values of the returned arguments
            will be such that a minimal amount of weight will be allocated when used.
            This will speed up the instantiation of the module, useful if weights
            will be overwritten in any case after the module instantiation.
    Returns:

    """
    if isinstance(mod, nn.Linear):
        kwargs = get_linear_args(mod, avoid_weight_allocation=avoid_weight_allocation)
    elif isinstance(mod, nn.LayerNorm):
        kwargs = get_layernorm_args(mod)
    elif isinstance(mod, nn.Embedding):
        kwargs = get_embedding_args(mod, avoid_weight_allocation=avoid_weight_allocation)
    else:
        raise ValueError

    kwargs["activation"] = act
    return kwargs


def quant_module(module, i, **quant_params):
    act, _ = get_act(module, i)
    modtype = module_map[type(module[i])]

    kwargs = get_module_args(module[i], act)
    new_module = modtype(**kwargs, **quant_params)
    new_module.weight.data = module[i].weight.data.clone()

    if module[i].bias is not None:
        new_module.bias.data = module[i].bias.data.clone()

    return new_module, i + int(bool(act)) + 1


def quantize_sequential(model, specials=None, tie_activation_quantizers=False, **quant_params):
    specials = specials or dict()

    i = 0
    quant_modules = []
    while i < len(model):
        if isinstance(model[i], QuantizedModule):
            quant_modules.append(model[i])
        elif type(model[i]) in module_map:
            new_module, new_i = quant_module(model, i, **quant_params)
            quant_modules.append(new_module)
            i = new_i
            continue

        elif type(model[i]) in specials:
            quant_modules.append(specials[type(model[i])](model[i], **quant_params))

        else:
            quant_modules.append(quantize_model(model[i], specials=specials, **quant_params))
        i += 1

    if isinstance(model, nn.ModuleList):
        return nn.ModuleList(quant_modules)
    else:
        return nn.Sequential(*quant_modules)


def quantize_model(
    model,
    specials: Optional[Dict[Type[nn.Module], Type[QuantizedModule]]] = None,
    tie_activation_quantizers: bool = False,
    **quant_params,
):
    """Process a model recursively, returning a quantized version of it.

    Each standard module in the model (like Linear, Conv2D, etc..) will be
    re-instantiated as a new quantized version of it, and the original weights
    will be assigned to it.

    All non-standard module will be processed recursively looking at its
    children, but the `specials` mapping can avoid this for specific module
    types, converting directly into a special quantized-version of that specific
    module.

    Args:
        model: the original model to be quantized.
        specials: A map from original modules to quantized version of the same.
        tie_activation_quantizers:
        **quant_params:

    Returns:
        quant_model
    """
    if specials is None:
        specials = {}

    if type(model) in specials:
        quant_model = specials[type(model)](model, **quant_params)

    elif type(model) in module_map.keys():
        avoid_weight_allocation = quant_params.get("avoid_weight_allocation", False)
        module_args = get_module_args(model, None, avoid_weight_allocation=avoid_weight_allocation)
        quantized_module_type = module_map[type(model)]
        quant_model = quantized_module_type(**module_args, **quant_params)
        quant_model.weight.data = model.weight.data

        if getattr(model, "bias", None) is not None:
            quant_model.bias.data = model.bias.data

    elif isinstance(model, nn.Sequential) or isinstance(model, nn.ModuleList):
        quant_model = quantize_sequential(
            model, specials, tie_activation_quantizers, **quant_params
        )

    else:
        # Unknown type, try to quantize all child modules
        quant_model = copy.deepcopy(model)
        for name, module in quant_model._modules.items():
            new_model = quantize_model(module, specials=specials, **quant_params)
            if new_model is not None:
                setattr(quant_model, name, new_model)

    return quant_model
