# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import warnings
from functools import partial, wraps

import click

from quantization.quantizers import QMethods
from quantization.range_estimators import OptMethod, RangeEstimators
from utils import DotDict
from utils.click_utils import ClickEnumOption, split_dict
from utils.enums import QuantSetup

click.option = partial(click.option, show_default=True)


def qat_options(func):
    @click.option(
        "--learn-ranges/--no-learn-ranges",
        is_flag=True,
        default=False,
        help="Learn quantization ranges.",
    )
    @click.option(
        "--fix-act-ranges/--no-fix-act-ranges",
        is_flag=True,
        default=False,
        help="Fix all activation quantization ranges during training",
    )
    @click.option(
        "--fix-weight-ranges/--no-fix-weight-ranges",
        is_flag=True,
        default=False,
        help="Fix all weight quantization ranges during training",
    )
    @click.option(
        "--quant-scales-lr-decrease",
        default=1.0,
        type=float,
        help="The Learning rate for quantization scales is multiplied by this number.",
    )
    @click.option(
        "--quant-scales-weight-decay",
        default=None,
        type=float,
        help="Weight decay on quant params if different from global weight decay.",
    )
    @click.option(
        "--quant-scales-domain",
        default="linear",
        type=click.Choice(["linear", "log"]),
        help="Print quantization ranges before and after training",
    )
    @click.option(
        "--print-quant-ranges",
        is_flag=True,
        help="Print quantization ranges before and after training",
    )
    @click.option("--print-parameters/--no-print-parameters", is_flag=True, default=False)
    @click.option(
        "--grad-scaling/--no-grad-scaling",
        is_flag=True,
        default=False,
        help="Do gradient scaling as in LSQ paper.",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "learn_ranges",
            "fix_act_ranges",
            "fix_weight_ranges",
            "quant_scales_lr_decrease",
            "quant_scales_weight_decay",
            "quant_scales_domain",
            "print_quant_ranges",
            "print_parameters",
            "grad_scaling",
        ]
        config.qat, remainder_kwargs = split_dict(kwargs, attrs)

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def quantization_options(func):
    @click.option(
        "--qmethod",
        type=ClickEnumOption(QMethods),
        required=True,
        help="Quantization scheme to use.",
    )
    @click.option(
        "--qmethod-act",
        type=ClickEnumOption(QMethods),
        default=None,
        help="Quantization scheme for activation to use. If not specified `--qmethod` " "is used.",
    )
    @click.option(
        "--weight-quant-method",
        default=RangeEstimators.current_minmax.name,
        type=ClickEnumOption(RangeEstimators),
        help="Method to determine weight quantization clipping thresholds.",
    )
    @click.option(
        "--weight-quant-block-size",
        default=None,
        type=int,
        help="Quantization block size for weight quantization. It is enable only if "
        "an integer number is provided. Only tested for fully connected layers",
    )
    @click.option(
        "--weight-opt-method",
        default=OptMethod.grid.name,
        type=ClickEnumOption(OptMethod),
        help="Optimization procedure for activation quantization clipping thresholds",
    )
    @click.option(
        "--num-candidates",
        type=int,
        default=None,
        help="Number of grid points for grid search in MSE range method.",
    )
    @click.option("--n-bits", default=8, type=int, help="Default number of quantization bits.")
    @click.option(
        "--n-bits-act", default=None, type=int, help="Number of quantization bits for activations."
    )
    @click.option(
        "--per-channel/--no-per-channel",
        is_flag=True,
        default=False,
        help="If given, quantize each channel separately.",
    )
    @click.option(
        "--freeze-bn/--no-freeze-bn",
        is_flag=True,
        default=False,
        help="If given, does not update BN when estimating quantization ranges.",
    )
    @click.option(
        "--percentile",
        type=float,
        default=None,
        help="Percentile clipping parameter (weights and activations)",
    )
    @click.option(
        "--act-quant/--no-act-quant",
        is_flag=True,
        default=True,
        help="Run evaluation with activation quantization or use FP32 activations",
    )
    @click.option(
        "--weight-quant/--no-weight-quant",
        is_flag=True,
        default=True,
        help="Run evaluation weight quantization or use FP32 weights",
    )
    @click.option(
        "--quant-setup",
        default=QuantSetup.all.name,
        type=ClickEnumOption(QuantSetup),
        help="Method to quantize the network.",
    )
    @click.option(
        "--freeze-integer-weights/--no-freeze-integer-weights",
        is_flag=True,
        default=False,
        help="Experimental feature: convert and store model weights as integer tensors during the PTQ. "
        "This will save space in ram, but training of the quantized model weights will be disabled.",
    )
    @click.option(
        "--avoid-weight-allocation/--force-weight-allocation",
        is_flag=True,
        default=False,
        help="Experimental feature: minimize/avoid weight allocation for Quantized version of the model "
        "during PTQ pipeline, given that these weights will be overridden few steps later "
        "by the original (not-quantized) model weights.",
    )
    @click.option(
        "--enable-quantized-weight-cache/--disable-quantized-weight-cache",
        is_flag=True,
        default=True,
        help="Enable the quantized-weight cache for QuantizedModel (i.e. in each QuantizedModule).",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "qmethod",
            "qmethod_act",
            "weight_quant_method",
            "weight_quant_block_size",
            "weight_opt_method",
            "num_candidates",
            "n_bits",
            "n_bits_act",
            "per_channel",
            "freeze_bn",
            "percentile",
            "act_quant",
            "weight_quant",
            "quant_setup",
            "freeze_integer_weights",
            "avoid_weight_allocation",
            "enable_quantized_weight_cache",
        ]
        config.quant, remainder_kwargs = split_dict(kwargs, attrs)

        config.quant.qmethod_act = config.quant.qmethod_act or config.quant.qmethod

        if not config.quant.weight_quant and config.quant.freeze_integer_weights:
            config.quant.freeze_integer_weights = False
            warnings.warn(
                f"The flag '--no-weight-quant' will disable '--freeze-integer-weights'. "
                f"To avoid this warning you can add the flag '--no-freeze-integer-weights'."
            )

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def activation_quantization_options(func):
    @click.option(
        "--act-quant-method",
        default=RangeEstimators.running_minmax.name,
        type=ClickEnumOption(RangeEstimators),
        help="Method to determine activation quantization clipping thresholds",
    )
    @click.option(
        "--act-opt-method",
        default=OptMethod.grid.name,
        type=ClickEnumOption(OptMethod),
        help="Optimization procedure for activation quantization clipping thresholds",
    )
    @click.option(
        "--act-num-candidates",
        type=int,
        default=None,
        help="Number of grid points for grid search in MSE/SQNR/Cross-entropy",
    )
    @click.option(
        "--act-momentum",
        type=float,
        default=None,
        help="Exponential averaging factor for running_minmax",
    )
    @click.option(
        "--cross-entropy-layer",
        default=None,
        type=str,
        help="Cross-entropy for activation range setting (often valuable for last layer)",
    )
    @click.option(
        "--num-est-batches",
        type=int,
        default=1,
        help="Number of training batches to be used for activation range estimation",
    )
    @click.option(
        "--act-quant-pattern",
        default=None,
        type=str,
        help="If given, only act quantizers matching this pattern are active, others "
        "will be removed (EXPERIMENTAL feature, mainly for Llama right now).",
    )
    @wraps(func)
    def func_wrapper(
        config,
        act_quant_method,
        act_opt_method,
        act_num_candidates,
        act_momentum,
        cross_entropy_layer,
        num_est_batches,
        act_quant_pattern,
        *args,
        **kwargs,
    ):
        config.act_quant = DotDict()
        config.act_quant.quant_method = act_quant_method
        config.act_quant.cross_entropy_layer = cross_entropy_layer
        config.act_quant.num_batches = num_est_batches
        config.act_quant.quant_pattern = act_quant_pattern

        config.act_quant.options = {}

        if act_num_candidates is not None:
            if act_quant_method not in (RangeEstimators.MSE,):
                raise ValueError("Wrong option num_candidates passed")
            else:
                config.act_quant.options["num_candidates"] = act_num_candidates

        if act_momentum is not None:
            if act_quant_method != RangeEstimators.running_minmax:
                raise ValueError("Wrong option momentum passed")
            else:
                config.act_quant.options["momentum"] = act_momentum

        if act_opt_method != OptMethod.grid:
            config.act_quant.options["opt_method"] = act_opt_method

        return func(config, *args, **kwargs)

    return func_wrapper


def qat_params(config):
    params = val_qparams(config)
    params["scale_domain"] = config.qat.quant_scales_domain

    # Range learning/estimation/fixed checks
    if config.quant.weight_quant and not config.quant.act_quant and config.qat.learn_ranges:
        assert (
            not config.qat.fix_weight_ranges
        ), "Choose either learn-ranges or fix ranges for weight only QAT"

    if config.quant.act_quant and not config.quant.weight_quant and config.qat.learn_ranges:
        assert (
            not config.qat.fix_act_ranges
        ), "Choose either learn-ranges or fix ranges for activations only QAT"

    if config.qat.learn_ranges:
        assert not (
            config.qat.fix_weight_ranges and config.qat.fix_act_ranges
        ), "You cannot have both weight and activation ranges fixed when learning ranges"

    return params


def val_qparams(config):
    weight_range_options = {}
    if config.quant.weight_quant_method in (RangeEstimators.MSE,):
        weight_range_options = dict(opt_method=config.quant.weight_opt_method)

    if config.quant.num_candidates is not None:
        weight_range_options["num_candidates"] = config.quant.num_candidates

    params = {
        "method": config.quant.qmethod.cls,
        "n_bits": config.quant.n_bits,
        "n_bits_act": config.quant.n_bits_act,
        "act_method": config.quant.qmethod_act.cls,
        "per_channel_weights": config.quant.per_channel,
        "percentile": config.quant.percentile,
        "quant_setup": config.quant.quant_setup,
        "weight_range_method": config.quant.weight_quant_method.cls,
        "weight_range_options": weight_range_options,
        "act_range_method": config.act_quant.quant_method.cls,
        "act_range_options": config.act_quant.options,
        "quantize_input": config.quant.quant_setup == QuantSetup.LSQ_input,
        "weight_block_size": config.quant.weight_quant_block_size,
        "avoid_weight_allocation": config.quant.avoid_weight_allocation,
    }
    params["weight_quant_kwargs"] = {}
    params["act_quant_kwargs"] = {}
    return params
