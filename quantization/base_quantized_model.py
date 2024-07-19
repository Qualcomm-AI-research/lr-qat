# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import logging
import warnings
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn

from quantization.base_quantized_classes import (
    QuantizedModule,
    _set_layer_estimate_ranges,
    _set_layer_estimate_ranges_train,
    _set_layer_fix_ranges,
    _set_layer_learn_ranges,
)
from quantization.hijacker import QuantizationHijacker
from quantization.quantizers.base_quantizers import QuantizerBase


class QuantizedModel(nn.Module):
    """
    Parent class for a quantized model. This allows you to have convenience functions to put the
    whole model into quantization or full precision or to freeze BN. Otherwise it does not add any
    further functionality, so it is not a necessity that a quantized model uses this class.
    """

    def __init__(self, input_size=(1, 3, 224, 224)):
        """
        Parameters
        ----------
        input_size:     Tuple with the input dimension for the model (including batch dimension)
        """
        super().__init__()
        self.input_size = input_size

    @property
    def device(self):
        return next(self.parameters()).device

    def load_state_dict(
        self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True
    ):
        """
        This function overwrites the load_state_dict of nn.Module to ensure that quantization
        parameters are loaded correctly for quantized model.

        """
        quant_state_dict = {
            k: v for k, v in state_dict.items() if k.endswith("_quant_a") or k.endswith("_quant_w")
        }

        if quant_state_dict:
            # Case 1: the quantization states are stored in the state_dict
            super().load_state_dict(quant_state_dict, strict=False)

        else:
            # Case 2 (older models): the quantization states are NOT stored in the state_dict but
            # only the scale factor _delta.
            warnings.warn(
                "Old state_dict without quantization state included. Checking for " "_delta instead"
            )
            # Add quantization flags to the state_dict
            for name, module in self.named_modules():
                if isinstance(module, QuantizedModule):
                    state_dict[".".join((name, "_quant_a"))] = torch.BoolTensor([False])
                    state_dict[".".join((name, "_quant_w"))] = torch.BoolTensor([False])
                    if (
                        ".".join((name, "activation_quantizer", "quantizer", "_delta"))
                        in state_dict.keys()
                    ):
                        module.quantized_acts()
                        state_dict[".".join((name, "_quant_a"))] = torch.BoolTensor([True])
                    if (
                        ".".join((name, "weight_quantizer", "quantizer", "_delta"))
                        in state_dict.keys()
                    ):
                        module.quantized_weights()
                        state_dict[".".join((name, "_quant_w"))] = torch.BoolTensor([True])

        # Load state dict
        super().load_state_dict(state_dict, strict)

    def set_caching(self, value: bool):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.caching = value

        self.apply(_fn)

    def disable_caching(self):
        self.set_caching(False)

    def enable_caching(self):
        self.set_caching(True)

    def quantized_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_weights()

        self.apply(_fn)

    def full_precision_weights(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_weights()

        self.apply(_fn)

    def quantized_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized_acts()

        self.apply(_fn)

    def full_precision_acts(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision_acts()

        self.apply(_fn)

    def quantized(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.quantized()

        self.apply(_fn)

    def full_precision(self):
        def _fn(layer):
            if isinstance(layer, QuantizedModule):
                layer.full_precision()

        self.apply(_fn)

    def cached_full_precision(self):
        quantizer_status = {}

        def the_func(layer):
            if isinstance(layer, QuantizedModule):
                quantizer_status[layer] = layer.get_quantizer_status()
                layer.full_precision()

        self.apply(the_func)
        return quantizer_status

    def requantize(self, quantizer_status):
        def the_func(layer):
            if layer in quantizer_status:
                layer.set_quantizer_status(quantizer_status[layer])

        self.apply(the_func)

    def clear_cache(self):
        def the_func(module):
            if isinstance(module, QuantizedModule):
                module.clear_cache()

        self.apply(the_func)

    # Methods for switching quantizer quantization states
    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)

    def fix_act_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
                _set_layer_fix_ranges(module.activation_quantizer)

        self.apply(_fn)

    def fix_weight_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "weight_quantizer"):
                _set_layer_fix_ranges(module.weight_quantizer)

        self.apply(_fn)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_act_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
                _set_layer_estimate_ranges(module.activation_quantizer)

        self.apply(_fn)

    def estimate_weight_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "weight_quantizer"):
                _set_layer_estimate_ranges(module.weight_quantizer)

        self.apply(_fn)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

    def reset_act_ranges(self):
        def _fn(module):
            if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
                module.activation_quantizer.reset_ranges()

        self.apply(_fn)

    def set_quant_state(
        self,
        weight_quant: Optional[bool] = None,
        act_quant: Optional[bool] = None,
    ):
        if act_quant is not None and act_quant:
            self.quantized_acts()
        elif act_quant is not None and not act_quant:
            self.full_precision_acts()

        if weight_quant is not None and weight_quant:
            self.quantized_weights()
        elif weight_quant is not None and not weight_quant:
            self.full_precision_weights()

    def set_quant_state_filtered(
        self,
        weight_quant: Optional[bool] = None,
        act_quant: Optional[bool] = None,
        filter: Optional[Union[List[str], str]] = None,
    ):
        if filter is None:
            return self.set_quant_state(weight_quant, act_quant)

        logging.info(f"All possible keys that can be selected with filters for current model:")
        for key in dict(self.named_modules()).keys():
            logging.info(key)
        logging.info("-" * 120)

        filters = [filter] if isinstance(filter, str) else filter
        modules = {}
        for filter in filters:
            new_modules = {name: module for name, module in self.named_modules() if name == filter}
            modules = {**modules, **new_modules}

        if len(modules) == 0:
            raise ValueError(f"No modules found to be quantized using filter: {filter}.")

        logging.info(
            f"Applying quant state (weight_quant={weight_quant}, act_quant={act_quant}) "
            f"to the following quantized modules:"
        )

        for filtered_module_name, filtered_module in modules.items():
            logging.info(f" - {filtered_module_name}")
            for name, m in filtered_module.named_modules():
                if not isinstance(m, QuantizedModule):
                    continue
                if weight_quant:
                    m.quantized_weights()
                else:
                    m.full_precision_weights()

                if act_quant:
                    m.quantized_acts()
                else:
                    m.full_precision_acts()

    def grad_scaling(self, grad_scaling=True):
        def _fn(module):
            if isinstance(module, QuantizerBase):
                module.grad_scaling = grad_scaling

        self.apply(_fn)

    def freeze_integer_weights(self, device=None):
        def _fn(module):
            if isinstance(module, QuantizationHijacker):
                module.freeze_integer_weights(device)

        self.apply(_fn)
