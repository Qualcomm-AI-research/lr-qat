# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import logging
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import Module

from quantization.base_quantized_model import QuantizedModel
from quantization.hijacker import QuantizationHijacker
from quantization.quantization_manager import QuantizationManager
from quantization.quantizers.base_quantizers import QuantizerBase
from utils import DotDict
from utils.enums import LR_QAT_PHI, LR_QAT_Method, TrainableParamGroup
from utils.quant_click_options import qat_params
from utils.quant_utils import (
    get_quant_model,
    pass_data_for_range_estimation,
    set_range_estimators,
)


def _print_parameters(model, logger):
    logger.info(f"Parameter shapes:")
    for name, param in model.named_parameters():
        logger.info(f"{name} -> {param.shape}")


def _print_quant_ranges(model, logger):
    logger.info("Quantization ranges before training:")
    for name, module in model.named_modules():
        if isinstance(module, QuantizationManager):
            from quantization.quantizers import DynamicAsymmetricUniformQuantizer

            if isinstance(module.quantizer, DynamicAsymmetricUniformQuantizer):
                msg = f"{name}: state={module.state}, no delta (dynamic quantizer)"
                logger.info(msg)
            else:
                delta = (
                    module.quantizer.delta if module.quantizer.is_initialized else "not initialized"
                )
                msg = f"{name}: state={module.state}, delta={delta}"
                logger.info(msg)


def separate_quantized_model_params(quant_model):
    """This method separates the parameters of the quantized model to into quantization and model
    parameters.

    Parameters
    ----------
    quant_model: QuantizedModel

    Returns
    -------
    quant_params: OrderedDict[str] -> torch.Tensor
    model_params: OrderedDict[str] -> torch.Tensor
    """
    quant_params = OrderedDict()
    model_params = OrderedDict()

    for mod_name, module in quant_model.named_modules():
        if isinstance(module, QuantizerBase):
            # use recurse=False to avoid adding the gradient estimator parameters to the
            # quantization parameters list
            for name, param in module.named_parameters(recurse=False):
                # delta and zero_float
                quant_param_name = ".".join((mod_name, name))
                quant_params[quant_param_name] = param

    def tensor_in_list(tensor, lst):
        return any([e is tensor for e in lst])

    found_params = quant_params.values()

    for n, p in quant_model.named_parameters():
        if not tensor_in_list(p, found_params):
            model_params[n] = p

    assert len(quant_params) + len(model_params) == len(
        list(quant_model.parameters())
    ), "{}; {} -- {}".format(
        len(model_params), len(quant_params), len(list(quant_model.parameters()))
    )
    return quant_params, model_params


def qat_get_quantized_model(
    config: DotDict,
    model: Module,
) -> QuantizedModel:
    qparams = qat_params(config)
    quant_model = get_quant_model(
        fp_model=model,
        model_type=config.model.model_type,
        qparams=qparams,
        act_quant_pattern=config.act_quant.quant_pattern,
    )
    quant_model.set_quant_state_filtered(
        weight_quant=config.quant.weight_quant,
        act_quant=config.quant.act_quant,
        filter=config.llm_extra.module_quant_filter,
    )
    return quant_model


def qat_prepare_model(config, model, train_dataloader, logger):
    # important in order to reduce memory requirements
    model.disable_caching()
    model.clear_cache()

    # attach and init LoRA parameters (if using SVD init)
    if (
        config.training.param_group in TrainableParamGroup.LORA
        and config.llm_extra.use_lora_svd_init
    ):
        _attach_and_init_quantized_lora(config, model, logger)

    # Range estimation
    pass_data_for_range_estimation(
        loader=train_dataloader,
        model=model,
        act_quant=config.quant.act_quant,
        weight_quant=config.quant.weight_quant,
        max_num_batches=config.act_quant.num_batches,
        use_fp16=config.base.use_fp16,
        use_bf16=config.base.use_bf16,
        # We set the quant state before and we don't wont to overwrite these settings.
        skip_set_quant_state=True,
    )

    # Put quantizers in desirable state
    set_range_estimators(config, model, skip_set_quant_state=True)

    # attach and init LoRA parameters (if not using SVD init)
    if (
        config.training.param_group in TrainableParamGroup.LORA
        and not config.llm_extra.use_lora_svd_init
    ):
        _attach_and_init_quantized_lora(config, model, logger)

    # Freeze int weights, if needed
    if config.quant.freeze_integer_weights:
        logger.info("Freezing integer weights")
        model.freeze_integer_weights()

    # Double check if we have the right parameters
    if config.qat.print_parameters:
        _print_parameters(model, logger)

    # print ranges before training, if needed
    if config.qat.print_quant_ranges:
        _print_quant_ranges(model, logger)

    return model


def _attach_and_init_quantized_lora(config, model, logger, use_lora_int_domain=False):
    # >> replace layers with LoRA counter-parts
    lora_kw = DotDict(
        r=config.llm_extra.lora_r,
        lora_alpha=config.llm_extra.lora_alpha,
        lora_dropout=config.llm_extra.lora_dropout,
        lora_method=config.llm_extra.lora_method,
        lora_b_std=config.llm_extra.lora_b_std,
        lora_save_int_tensors=config.llm_extra.lora_save_int_tensors,
        use_checkpointing=config.llm_extra.use_checkpointing,
        use_lora_svd_init=config.llm_extra.use_lora_svd_init,
        lora_svd_init_steps=config.llm_extra.lora_svd_init_steps,
        use_bf16_ab=config.llm_extra.use_bf16_ab,
    )

    # specify output dir for saving
    if config.llm_extra.lora_save_int_tensors:
        lora_kw.output_dir = config.base.output_dir

    # Llama, Mistral
    if config.model.model_type in ("llama", "mistral"):
        for layer_idx, layer in enumerate(model.get_attention_blocks()):
            # Q, K, V, O
            self_attn = layer.self_attn
            _init_quantized_lora_fn(self_attn.q_proj, f"layer.{layer_idx}.q_proj", lora_kw)
            _init_quantized_lora_fn(self_attn.k_proj, f"layer.{layer_idx}.k_proj", lora_kw)
            _init_quantized_lora_fn(self_attn.v_proj, f"layer.{layer_idx}.v_proj", lora_kw)
            _init_quantized_lora_fn(self_attn.o_proj, f"layer.{layer_idx}.o_proj", lora_kw)

            # MLP
            mlp = layer.mlp
            _init_quantized_lora_fn(mlp.gate_proj_act, f"layer.{layer_idx}.gate_proj_act", lora_kw)
            _init_quantized_lora_fn(mlp.up_proj, f"layer.{layer_idx}.up_proj", lora_kw)
            _init_quantized_lora_fn(mlp.down_proj, f"layer.{layer_idx}.down_proj", lora_kw)

    else:
        raise ValueError(f"model type '{config.model.model_type}' is not supported")


def _init_quantized_lora_fn(m: QuantizationHijacker, name: str, lora_kw: DotDict):
    # if we are not applying weight quantization on this module, we do not apply quantized lora.
    if not all(m._quant_w):
        return

    if lora_kw.lora_method in (LR_QAT_Method.lr_qat_int, *LR_QAT_PHI):
        # We have to move LoRA adapter values to the quantization domain
        use_lora_int_domain = True
    elif lora_kw.lora_method == LR_QAT_Method.naive:
        use_lora_int_domain = False
    else:
        raise ValueError(
            f"Lora method {lora_kw.lora_method} is unknown to "
            f"`qat_utils._init_quantized_lora_fn()`."
        )

    r = lora_kw.r
    m.lora_r = r
    m.lora_method = lora_kw.lora_method
    m.use_checkpointing = lora_kw.use_checkpointing

    m.weight.requires_grad = False

    if r <= 0:  # we still wanna keep it in case we want to use checkpointing for scales only
        m.lora_scaling = 1
        return

    m.lora_scaling = lora_kw.lora_alpha / r

    in_features = m.in_features
    out_features = m.out_features
    dtype = torch.float32
    A = torch.zeros(out_features, r, dtype=dtype, device=m.weight.device, requires_grad=True)
    Bt = torch.zeros(r, in_features, dtype=dtype, device=m.weight.device, requires_grad=True)
    m.lora_Bt = nn.Parameter(Bt)
    m.lora_A = nn.Parameter(A)

    with torch.no_grad():
        # Init
        nn.init.kaiming_uniform_(m.lora_Bt, a=math.sqrt(5))
        if lora_kw.lora_b_std == 0.0:
            # standard LoRA init
            nn.init.zeros_(m.lora_A)
        else:
            nn.init.normal_(m.lora_A, std=lora_kw.lora_b_std)

        if lora_kw.use_lora_svd_init and lora_kw.lora_svd_init_steps > 0:
            W = m.weight.data
            m.weight_quantizer.reset_ranges()
            m.weight_quantizer.estimate_ranges()

            Q, A, B, err = init_lora_svd(
                W=W,
                w_quant=m.weight_quantizer,
                A=m.lora_A.data,
                B=m.lora_Bt.data.t(),
                rank=r,
                scaling=m.lora_scaling,
                steps=lora_kw.lora_svd_init_steps,
            )

            if use_lora_int_domain:
                # We have to move LoRA adapter to the quantization domain,
                # so we quantize A and B values dividing by the scale:
                # (sym quant only)
                if out_features == in_features:
                    A /= torch.sqrt(m.weight_quantizer.quantizer.scale)
                    B /= torch.sqrt(m.weight_quantizer.quantizer.scale)
                else:
                    A /= m.weight_quantizer.quantizer.scale

            m.lora_A.data = A
            m.lora_Bt.data = B.t()
            m.weight.data = Q
            m.freeze_integer_weights()

        if lora_kw.lora_save_int_tensors:
            m.output_dir = lora_kw.output_dir
            m.name = name

    if lora_kw.use_bf16_ab:
        m.lora_Bt.data = m.lora_Bt.data.to(torch.bfloat16)
        m.lora_A.data = m.lora_A.data.to(torch.bfloat16)


def init_lora_svd(
    W: torch.Tensor,
    w_quant: QuantizationManager,
    A: torch.Tensor,
    B: torch.Tensor,
    rank: int,
    scaling: float = 1.0,
    steps: int = 1,
    early_stop=False,
):
    if steps < 1:
        return W, A, B

    def objective(W, Q, C):
        return torch.norm(W - (w_quant(Q) + C), p="fro")

    def compute_C(A, B):
        return (A @ B.t()) * scaling

    w_quant.reset_ranges()
    w_quant.estimate_ranges()
    Q = w_quant(W)
    w_quant.fix_ranges()
    A.zero_()
    B.zero_()
    C = compute_C(A, B)

    best_err = init_err = objective(W, Q, C)
    best_params = (Q, A, B, C)
    best_step = 0

    for t in range(1, steps + 1):
        w_quant.reset_ranges()
        w_quant.estimate_ranges()
        Q = w_quant(W - C)
        w_quant.fix_ranges()

        R = W - Q
        U, S, V = torch.svd(R / scaling)

        sqrtS = torch.sqrt(S)[None, :rank]
        A = U[:, :rank] * sqrtS
        B = V[:, :rank] * sqrtS
        C = compute_C(A, B)

        err = objective(W, Q, C)
        if err < best_err:
            best_step = t
            best_err = err
            best_params = (Q, A, B, C)

    logging.info(
        f"Init Step  (0)  - Target error: {init_err}\n"
        f"Final step ({t}) - Target error: {err}\n"
        f"Best step  ({best_step}) - Target error: {best_err}\n"
    )
    if early_stop:
        Q, A, B, C = best_params
        err = best_err

    w_quant.reset_ranges()
    w_quant.estimate_ranges()
    w_quant(W - C)
    w_quant.fix_ranges()

    return Q, A, B, err
