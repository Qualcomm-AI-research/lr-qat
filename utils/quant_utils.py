# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
from typing import List, Optional

import numpy.typing as npt
import torch
from torch import nn

from models.quantized_llama import QuantizedLlamaForCausalLM
from models.quantized_mistral import QuantizedMistralForCausalLM
from quantization.base_quantized_classes import FP32Acts, QuantizedModule
from quantization.base_quantized_model import QuantizedModel
from quantization.quantization_manager import QuantizationManager
from quantization.quantizers.base_quantizers import QuantizerBase
from utils import convert_transformer_float_input

conf_to_act_quantizers_map = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
    "o": "o_proj",
    "ke": "attn_key_act_quantizer",  # key embeddings (key+embeddings)
    "qe": "attn_query_act_quantizer",  # query embeddings (query+embeddings)
    "aw": "attn_softmax_inp_quantizer",  # attention weights (pre softmax)
    "ap": "attn_probs_act_quantizer",  # attention probabilities (post softmax)
    "ao": "attn_act_re_quantizer",  # attention output (after att_weight*values)
    "g": "gate_proj_act",
    "d": "down_proj",
    "u": "up_proj",
    "mm": "mlp_mul_act_quantizer",  # MLP elementwise multiplication
    "na": "attn_ln_act_quantizer",  # layer norm before attention
    "nm": "mlp_ln_act_quantizer",  # layer norm before mlp
    "ra": "attn_res_act_quantizer",  # residual add after attention
    "rm": "mlp_res_act_quantizer",  # residual add after mlp
}

act_quant_to_conf_map = {v: k for k, v in conf_to_act_quantizers_map.items()}


def apply_custom_act_quant_pattern(model, act_quant_pattern):
    if not isinstance(model, QuantizedLlamaForCausalLM):
        raise NotImplementedError(
            f"Custom activation quantization pattern is currently only "
            f"supported for Llama ({type(model).__name__} is not supported)."
        )

    # We don't wanna consider embedding quantization for now
    model.model.embed_act_quantizer = FP32Acts()

    # Check all activation quantizers whether we wanna have them
    print(f"\n\nRemove activation quantizers not in config: {act_quant_pattern}\n")

    # Convert config, split if bit width is given as {name}={bit}
    act_quant_config = {}
    for quant_conf in act_quant_pattern.split("."):
        quant_conf = quant_conf.split("=")
        act_quant_config[quant_conf[0]] = int(quant_conf[1]) if len(quant_conf) == 2 else None

    for name, module in model.named_modules():
        if isinstance(module, QuantizedModule) and hasattr(module, "activation_quantizer"):
            # We have a module holding an activation quantizer
            mname = name.split(".")[-1]
            if act_quant_to_conf_map[mname] not in act_quant_config:
                print(f"-- {mname} not in config, remove activatiton quantizer")
                module.activation_quantizer = FP32Acts()
            else:
                new_bitwidth = act_quant_config[act_quant_to_conf_map[mname]]
                if new_bitwidth is not None:
                    module.activation_quantizer.quantizer.n_bits = new_bitwidth
                    print(f"Keep act quantizer {mname}, set to {new_bitwidth} bits")
                else:
                    print(f"Keep act quantizer {mname}")

    # print all activation quantizers
    print("\n\nAll remaining activation quantizers:\n")
    for name, module in model.named_modules():
        if isinstance(module, QuantizationManager):
            if "weight" not in name:
                print(name)


def get_quant_model(fp_model, model_type, qparams, act_quant_pattern=None) -> QuantizedModel:
    """
    Convert FP model into a quantized model.

    Parameters
    ----------
    fp_model: nn.Module
        FP model.

    model_type: str
        HuggingFace model type.

    qparams: dict
        Quantization parameters.

    Returns
    -------
    model: QuantizedModel
        Quantized model.
    """
    if model_type == "llama":
        model = QuantizedLlamaForCausalLM(fp_model, **qparams)
    elif model_type == "mistral":
        model = QuantizedMistralForCausalLM(fp_model, **qparams)
    else:
        raise NotImplementedError(f"Quantized {model_type} is not supported")

    if act_quant_pattern is not None:
        apply_custom_act_quant_pattern(model, act_quant_pattern)

    return model


def try_move_to_device(tensor: torch.Tensor, device: Optional[torch.device]):
    """Move input tensor to the target device if the device is valid.

    Args:
        tensor: input tensor.
        device: target device, if the device is a 'meta' device from accelerate or None, tensor will not be moved.

    Returns: the tensor, possibly moved to the target device.

    """
    if device is not None and device.type != "meta":
        return tensor.to(device)
    return tensor


def pass_data_for_range_estimation(
    loader: torch.utils.data.DataLoader,
    model: QuantizedModel,
    act_quant: Optional[bool] = None,
    weight_quant: Optional[bool] = None,
    max_num_batches: int = 20,
    inp_idx: int = 0,
    use_fp16: bool = False,
    use_bf16: bool = False,
    skip_set_quant_state: bool = False,
) -> List[npt.ArrayLike]:
    """

    Parameters
    ----------
    loader: torch.utils.data.DataLoader
    model: QuantizedModel
        Quantized model
    act_quant: bool
        activation quantization (optional), otherwise use the current state of the quantize model
    weight_quant: bool
        weights quantization (optional), otherwise use the current state of the quantize model
    max_num_batches: int (20)
        number of batches for activation range estimation
    cross_entropy_layer: Optional[str], None by default
        name of cross entropy layer
    inp_idx: int (0)
        data index when the output of dataloader is a tuple
    use_fp16: bool (False)
        convert data to fp16

    Returns
    -------
    batches (List[np.array]): either an empty list or list of numpy arrays
    """
    print("\nEstimate quantization ranges on training data")
    if not skip_set_quant_state:
        model.set_quant_state(weight_quant, act_quant)

    model.eval()

    batches = []
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for i, data in enumerate(loader):
            if isinstance(data, (tuple, list)):
                x = try_move_to_device(data[inp_idx], model_device)
                batches.append(x.data.cpu().numpy())
                model(x)

            elif isinstance(data, dict) and isinstance(inp_idx, str):
                x = try_move_to_device(data["image"], model_device)
                model(x)

            else:
                x = {k: try_move_to_device(v, model_device) for k, v in data.items()}
                x = convert_transformer_float_input(x, bf16=use_bf16, fp16=use_fp16)
                model(**x)

            print(f"processed step={i}")
            if i >= max_num_batches - 1 or act_quant is False:
                break
    return batches


def set_range_estimators(config, model, skip_set_quant_state: bool = True):
    if config.qat.learn_ranges:
        print("Make quantizers learnable")
        model.learn_ranges()
    else:
        print("Quantization ranges are updated")
        model.estimate_ranges_train()  # we use updating ranges in training as default

    # Freeze quantization ranges if applicable
    if config.qat.fix_weight_ranges:
        print("Fix Weight quantization ranges")
        model.fix_weight_ranges()

    if config.qat.fix_act_ranges:
        print("Fix Activation quantization ranges")
        model.fix_act_ranges()

    if config.qat.grad_scaling:
        print("Activate gradient scaling")
        model.grad_scaling(True)

    if not skip_set_quant_state:
        # Ensure we have the desired quant state
        model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)


def print_quantizer_stats(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantizationManager):
            print(
                name,
                module.state,
                module.quantizer.delta if module.quantizer.is_initialized else "- not initialized",
            )
