# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import logging
from typing import Optional

import accelerate
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from quantization.base_quantized_model import QuantizedModel
from utils import DotDict, Stopwatch, get_and_log_cuda_memory
from utils.quant_click_options import val_qparams
from utils.quant_utils import get_quant_model, pass_data_for_range_estimation


def ptq_main(
    config: DotDict,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    logger: Optional[logging.Logger] = None,
    accelerator: Optional[accelerate.Accelerator] = None,
) -> QuantizedModel:
    """Apply PTQ to a given model.

    Args:
        config: configuration DotDict.
        model: model resident on CPU.
        train_dataloader: train dataloader that will be used for PTQ range estimation and adaround.
        logger:
        accelerator: if not None, the model and dataloader will be prepared with the Accelerator to move data in a
            specific device.
        cpu_offload: if true, the model will be automatically offloaded to CPU and weights will be automatically
            loaded on GPU memory when used.

    Returns: a quantized model and the train dataloader.
    """
    if logger is None:
        logger = logging.getLogger()

    with Stopwatch("ptq_get_quantized_model") as s:
        model = ptq_get_quantized_model(config, model)

    logger.debug("Quantized model:")
    logger.debug(model)

    if config.quant.weight_quant and config.quant.freeze_integer_weights:
        with Stopwatch("freeze_integer_weights") as s:
            model.freeze_integer_weights(accelerator.device)
    else:
        logger.info("Freeze integer weights: SKIPPED")

    # NB: We don't want to move the model to GPU *BEFORE* ptq_get_quantized_model,
    # to avoid GPU memory leaks
    if accelerator is not None:
        train_dataloader = accelerator.prepare(train_dataloader)
        model = accelerator.prepare(model)

    model = ptq_apply_range_estimation(config, model, train_dataloader)
    get_and_log_cuda_memory(logger, "After range estimation")
    return model


def ptq_get_quantized_model(config: DotDict, model: Module) -> QuantizedModel:
    qparams = val_qparams(config)
    quant_model = get_quant_model(
        fp_model=model,
        model_type=config.model.model_type,
        qparams=qparams,
        act_quant_pattern=config.act_quant.quant_pattern,
    )
    quant_model.set_quant_state(
        weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant
    )
    model.eval()
    if not config.quant.enable_quantized_weight_cache:
        quant_model.disable_caching()
    return quant_model


def ptq_apply_range_estimation(
    config: DotDict,
    model_quant: QuantizedModel,
    train_dataloader: DataLoader,
) -> QuantizedModel:

    pass_data_for_range_estimation(
        loader=train_dataloader,
        model=model_quant,
        act_quant=config.quant.act_quant,
        weight_quant=config.quant.weight_quant,
        max_num_batches=config.act_quant.num_batches,
        use_fp16=config.base.use_fp16,
        use_bf16=config.base.use_bf16,
    )

    # model put into the correct state
    model_quant.fix_ranges()
    model_quant.set_quant_state(
        weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant
    )
    return model_quant
