#!/usr/bin/env python
# coding=utf-8

# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""
The main script for pre-training, fine-tuning and validating LLMs on Causal Language Modeling (CLM)
task.
"""
import json
import logging
import math
import os
from collections import OrderedDict
from functools import partial
from pprint import pformat
from typing import Optional

import click
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import get_scheduler

from lmeval.click_options import lm_eval_options
from lmeval.evaluate import lm_eval
from quantization.hijacker import QuantizationHijacker
from utils import (
    DotDict,
    Stopwatch,
    attach_act_hooks,
    get_and_log_cuda_memory,
    truncate_batch_to_block_size,
)
from utils.enums import (
    BaseEnumOptions,
    DatasetSetup,
    TrainableParamGroup,
    ValidationQuantMethod,
)
from utils.huggingface_utils import (
    evaluate,
    init_accelerator_and_logger,
    load_and_tokenize_datasets,
    make_dataloader,
    make_model_and_tokenizer,
)
from utils.llm_click_options import (
    llm_combined_base_options,
    llm_extra_options,
    llm_quant_options,
    llm_training_options,
)
from utils.ptq_utils import ptq_apply_range_estimation, ptq_main
from utils.qat_utils import (
    qat_get_quantized_model,
    qat_prepare_model,
    separate_quantized_model_params,
)
from utils.quant_click_options import (
    activation_quantization_options,
    qat_options,
    quantization_options,
)
from utils.tb_utils import tb_log_histograms, tb_log_scalars

# (re-)enable fast matmuls using Tensor cores
# since PyTorch>=1.12, this option is automatically set to False (was True before)
torch.backends.cuda.matmul.allow_tf32 = True


# setup logger
logger = get_logger("CLM_main")
logger.setLevel(logging.INFO)


# setup click
class Config(DotDict):
    pass


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
def click_group():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


# show default values for all options
click.option = partial(click.option, show_default=True)


def _setup_optimizer(config, model, quant: bool = False):
    # move the model parameters to CPU s.t. the optimizer state is on CPU, if needed
    if config.training.cpu_opt_state:
        model.cpu()

    # split parameters into {model, quant} parameter groups
    if quant:
        quant_params, model_params = separate_quantized_model_params(model)

        if config.quant.freeze_integer_weights:
            quant_params = OrderedDict()
            for name, module in model.named_modules():
                if isinstance(module, QuantizationHijacker) and module.scale is not None:
                    quant_params[f"{name}.scale"] = module.scale

        if config.qat.fix_weight_ranges:
            quant_params = {}
    else:
        model_params = {n: p for n, p in model.named_parameters()}
        quant_params = {}

    # Define LoRA params (mutually exclusive with model_params)
    model_lora_params = {}

    # Exclude model parameters, if needed
    if config.training.param_group == TrainableParamGroup.all:
        pass

    elif config.training.param_group == TrainableParamGroup.none:
        model_params = {}

    elif config.training.param_group == TrainableParamGroup.freeze_embd:
        if not quant:
            raise ValueError(f"'freeze_embd' parameter group is only supported for quantized training")

        for e in model.get_embeddings():
            for k, v in model_params.items():
                if e.weight is v:
                    logger.info(f"Removing embedding {k} form the list of learnable parameters")
                    del model_params[k]
                    break

    elif config.training.param_group == TrainableParamGroup.norm_only:
        new_model_params = {}

        layernorm_keyword = "norm"
        # OPT -> 'layer_norm'
        # Llama -> 'layernorm'

        for name, p in model_params.items():
            if layernorm_keyword in name:
                new_model_params[name] = p

        model_params = new_model_params

    elif config.training.param_group in TrainableParamGroup.LORA:
        model_lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}

        if config.training.param_group == TrainableParamGroup.lora:
            model_params = {}

        elif config.training.param_group == TrainableParamGroup.lora_head:
            model_params = {}
            lm_head = model.get_head()
            for n, p in lm_head.named_parameters():
                model_params[n] = p

        elif config.training.param_group == TrainableParamGroup.lora_head_norm:
            # filter norm params
            new_model_params = {}
            layernorm_keyword = "norm"
            for name, p in model_params.items():
                if layernorm_keyword in name:
                    new_model_params[name] = p
            model_params = new_model_params

            # add head
            lm_head = model.get_head()
            for n, p in lm_head.named_parameters():
                model_params[n] = p

        elif config.training.param_group == TrainableParamGroup.lora_head_norm_embd:
            if not quant:
                raise ValueError(
                    f'Param group "lora_head_norm_embd" is only supported for quantized training'
                )

            # filter norm params
            new_model_params = {}
            layernorm_keyword = "norm"
            for name, p in model_params.items():
                if layernorm_keyword in name:
                    new_model_params[name] = p
            model_params = new_model_params

            # add head
            lm_head = model.get_head()
            for n, p in lm_head.named_parameters():
                model_params[n] = p

            # add embeddings
            for e in model.get_embeddings():
                for n, p in e.named_parameters():
                    model_params[n] = p
        else:
            raise ValueError(f'Unknown param group "{config.training.param_group}"')

    # Split (model) weights in two groups, one with weight decay and the other not.
    no_decay = ["bias"] if config.training.wd_ln_gamma else ["bias", "norm.weight"]

    # Group parameters for optimization
    optimizer_grouped_parameters = [
        # model params possibly with weight decay
        {
            "params": [p for n, p in model_params.items() if not any(x in n for x in no_decay)],
            "weight_decay": config.training.weight_decay,
        },
        # model params without weight decay
        {
            "params": [p for n, p in model_params.items() if any(x in n for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # LoRA A, B
    if model_lora_params:
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in model_lora_params.items()],
                "weight_decay": 0.0,
                "lr": config.training.lr_ab,
            }
        )

    # quant. parameters
    if quant:
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in quant_params.items()],
                "weight_decay": config.qat.quant_scales_weight_decay,
                "lr": config.training.scales_lr,
            }
        )

    # make optimizer
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )

    # move the model back to GPU in case we use optimizer offloading
    if config.training.cpu_opt_state:
        model.cuda()

    return optimizer


def _get_lr_scheduler(config, optimizer):
    num_warmup_steps = config.training.num_warmup_steps
    num_training_steps = config.training.max_train_steps

    if config.training.final_lr_fraction > 0:
        if config.training.lr_scheduler_type == "linear":
            # tweak number of steps for LR scheduler so that the final LR is not 0 but desired value
            warmup_ratio = num_warmup_steps / max(1.0, num_training_steps)
            a = 1 / (1 - (1.0 - warmup_ratio) * config.training.final_lr_fraction)
            num_warmup_steps = int(a * num_warmup_steps)
            num_training_steps = int(a * num_training_steps)
        else:
            raise ValueError(
                f'--final-lr-fraction is not supported for LR schedule "'
                f'{config.training.lr_scheduler_type}"'
            )

    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


def _eval_on_wikitext(config, accelerator, model, tokenizer, logger):
    config.data.dataset_setup = DatasetSetup.wikitext_2
    _, eval_dataset = load_and_tokenize_datasets(config, tokenizer, accelerator, logger)
    eval_dataloader = make_dataloader(config, eval_dataset, train=False)
    eval_dataloader = accelerator.prepare(eval_dataloader)
    metrics = evaluate(config, model, eval_dataloader, accelerator)
    return metrics


def _eval_extra_datasets(config, accelerator, model, tokenizer, logger):
    out = {}

    # ** Eval on Wikitext **
    if not config.data.dataset_setup in (
        DatasetSetup.wikitext_2,
    ):  # skip wikitext if we are already validating on it
        metrics = _eval_on_wikitext(config, accelerator, model, tokenizer, logger)
        logger.info(f"perplexity_wikitext: {metrics['perplexity']}")
        out["wikitext"] = metrics["perplexity"]

    # ===== Eval on 2048 seq length =====

    orig_block_size = config.model.block_size
    config.model.block_size = 2048

    # ** Eval on Wikitext **
    metrics = _eval_on_wikitext(config, accelerator, model, tokenizer, logger)
    logger.info(f"perplexity_wikitext_2048: {metrics['perplexity']}")
    out["wikitext_2048"] = metrics["perplexity"]

    # set block size to the original value
    config.model.block_size = orig_block_size

    return out


def _merge_lora_weights(config, model):
    logger.info("Fusing LoRA weights")

    # Llama, Mistral
    if config.model.model_type in ("llama", "mistral"):
        for layer_idx, layer in enumerate(model.get_attention_blocks()):
            # Q, K, V, O
            self_attn = layer.self_attn
            self_attn.q_proj.lora_merge_weights = True
            self_attn.k_proj.lora_merge_weights = True
            self_attn.v_proj.lora_merge_weights = True
            self_attn.o_proj.lora_merge_weights = True

            # MLP
            mlp = layer.mlp
            mlp.gate_proj_act.lora_merge_weights = True
            mlp.up_proj.lora_merge_weights = True
            mlp.down_proj.lora_merge_weights = True

    else:
        raise ValueError(f"model type '{config.model.model_type}' is not supported")


def _train(config: DotDict, quant: bool = False):
    # setup accelerator and loggers
    accelerator = init_accelerator_and_logger(config, logger)

    # log run options
    logger.info("Running with options:")
    logger.info(pformat(config))

    # set seed
    if config.base.seed is not None:
        logger.info(f"Setting the random seed to {config.base.seed}")
        set_seed(config.base.seed)

    # setup model and tokenizer
    model, tokenizer = make_model_and_tokenizer(config, logger)

    # log FP model
    logger.info("FP Model:")
    logger.info(model)
    get_and_log_cuda_memory(logger, "FP Model")

    # load datasets
    train_dataset, eval_dataset = load_and_tokenize_datasets(config, tokenizer, accelerator, logger)

    # make dataloaders
    train_dataloader = make_dataloader(config, train_dataset, train=True)
    eval_dataloader = make_dataloader(config, eval_dataset, train=False)

    if quant:
        # prepare model for QAT
        model = qat_get_quantized_model(config, model)
        logger.setLevel(logging.DEBUG)
        logger.debug("Quantized model:")
        logger.debug(model)

        # NB: We don't want to move the model to GPU *before* get_quant_model
        #     because some of the QuantizedModel constructor create memory leaks
        model = accelerator.prepare(model)
        model = qat_prepare_model(config, model, train_dataloader, logger)
        get_and_log_cuda_memory(logger, "After prepare model for QAT")

    elif config.training.param_group in TrainableParamGroup.LORA:
        raise RuntimeError(f"LoRA is only supported for train-quantized")

    else:
        model = accelerator.prepare(model)

    # setup optimizer
    optimizer = _setup_optimizer(config, model, quant=quant)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    if config.training.max_train_steps is None:
        config.training.max_train_steps = (
            config.training.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    # setup LR scheduler
    lr_scheduler = _get_lr_scheduler(config, optimizer)

    # Prepare the rest with our `accelerator`.
    optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may
    # have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.training.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        config.training.max_train_steps = (
            config.training.num_train_epochs * num_update_steps_per_epoch
        )

    # Afterwards we recalculate our number of training epochs
    config.training.num_train_epochs = math.ceil(
        config.training.max_train_steps / num_update_steps_per_epoch
    )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = config.base.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use
    # The trackers initializes automatically on the main process.
    if config.logging.with_tracking:
        accelerator.init_trackers("tb_logs")

    # Train!
    total_batch_size = (
        config.base.per_device_train_batch_size
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.training.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {config.base.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.training.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(config.training.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.base.resume_from_checkpoint:
        if (
            config.base.resume_from_checkpoint is not None
            or config.base.resume_from_checkpoint != ""
        ):
            accelerator.print(f"Resumed from checkpoint: {config.base.resume_from_checkpoint}")
            accelerator.load_state(config.base.resume_from_checkpoint)
            path = os.path.basename(config.base.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * config.training.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch

    # Load model state only
    elif config.model.model_state_dict_path is not None:
        logger.info(f'Loading model state dict from "{config.model.model_state_dict_path}"')

        state_dict = torch.load(config.model.model_state_dict_path)
        # skip lora
        new_state_dict = {}
        for name, value in state_dict.items():
            if not "lora_" in name:
                new_state_dict[name] = value
        model.load_state_dict(new_state_dict)

    # attach hooks for activation stats
    if config.logging.with_tracking and config.logging.tb_detailed_logging:
        act_dict = attach_act_hooks(model)

    # *** Evaluation (before training) ***
    s_eval = Stopwatch()
    perplexity_hist = []
    if config.llm_extra.eval_before_training:
        s_eval.start()

        metrics = evaluate(
            config,
            model,
            eval_dataloader,
            accelerator,
            max_num_batches=config.data.validation_num_batches,
        )
        logger.info(
            f"Before Training -- perplexity: {metrics['perplexity']} eval_loss: {metrics['eval_loss']}"
        )

        if config.logging.with_tracking:
            accelerator.log(
                {"perplexity": metrics["perplexity"], "eval_loss": metrics["eval_loss"], "step": 0},
                step=0,
            )
        s_eval.stop()

        # update history
        perplexity_hist.append(metrics["perplexity"])

    # ***** Training loop *****
    s_train = Stopwatch()
    running_train_loss = 0
    running_train_loss_hist = []
    mem_usage = {}
    extra_metrics = {}
    for epoch in range(starting_epoch, config.training.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            # truncate batch in case of Slimpajama
            if (
                config.data.dataset_setup
                in (DatasetSetup.slimpajama_wiki, DatasetSetup.slimpajama_wikitext_2)
                and config.model.block_size < 2048
            ):
                batch = truncate_batch_to_block_size(batch, config.model.block_size)

            # copy labels if not present
            if not "labels" in batch:
                batch["labels"] = batch["input_ids"].clone()

            # We need to skip steps until we reach the resumed step
            if config.base.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % config.training.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            # start/continue stopwatch
            s_train.start()

            # Optimizer step
            with accelerator.accumulate(model):
                outputs = model(**batch)

                # log and store cuda memory usage
                if step <= config.training.gradient_accumulation_steps:
                    out = get_and_log_cuda_memory(logger, "1_after_forward")
                    mem_usage.update(out)

                # update running train loss
                loss = outputs.loss
                loss_float = loss.detach().float().item()
                if running_train_loss == 0:
                    running_train_loss = loss_float
                else:
                    gamma = config.logging.running_train_loss_gamma
                    running_train_loss = gamma * running_train_loss + (1.0 - gamma) * loss_float

                # backward pass
                accelerator.backward(loss)

                # apply grad clipping
                if (
                    config.training.max_grad_norm is not None
                    and config.training.max_grad_norm > 0
                    and accelerator.sync_gradients
                ):
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        max_norm=config.training.max_grad_norm,
                        norm_type=config.training.grad_norm_type,
                    )

                if (
                    config.training.cpu_opt_state
                    and (step + 1) % config.training.gradient_accumulation_steps == 0
                ):
                    model.cpu()

                optimizer.step()

                if (
                    config.training.cpu_opt_state
                    and (step + 1) % config.training.gradient_accumulation_steps == 0
                ):
                    model.cuda()

                if not accelerator.optimizer_step_was_skipped:
                    # do not update LR if the grad update was skipped (because of overflow in grad
                    # computation cause by mixed-precision)
                    lr_scheduler.step()

                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                completed_steps += 1

                # update progress bar
                if completed_steps % config.logging.tqdm_update_interval == 0:
                    progress_bar.update(config.logging.tqdm_update_interval)

                # detailed logging in Tensorboard
                if config.logging.with_tracking and config.logging.tb_detailed_logging:
                    _tb_log_scalars(config, model, accelerator, completed_steps, act_dict, logger)
                    _tb_log_histograms(
                        config, model, accelerator, completed_steps, act_dict, logger
                    )

                # save model and state if needed
                if isinstance(checkpointing_steps, int):
                    if completed_steps > 0 and completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if config.base.output_dir is not None:
                            output_dir = os.path.join(config.base.output_dir, output_dir)
                        accelerator.save_state(output_dir)

            # pause/stop training stopwatch
            s_train.stop()

            # Eval after every N steps, if needed
            # (last two conditions: avoid re-evals at the last step and if grad acc steps > 1)
            if (
                config.llm_extra.eval_every_step is not None
                and config.llm_extra.eval_every_step > 0
                and completed_steps % config.llm_extra.eval_every_step == 0
                and completed_steps < config.training.max_train_steps
                and (step + 1) % config.training.gradient_accumulation_steps == 0
            ):
                s_eval.start()
                metrics = evaluate(
                    config,
                    model,
                    eval_dataloader,
                    accelerator,
                    max_num_batches=config.data.validation_num_batches,
                )
                s_eval.stop()

                logger.info(
                    f"Completed steps {completed_steps} (step {step}) -- perplexity: {metrics['perplexity']} "
                    f"eval_loss: {metrics['eval_loss']}"
                )

                # log metrics
                accelerator.log(
                    {
                        "running_train_loss": running_train_loss,
                        "perplexity": metrics["perplexity"],
                        "eval_loss": metrics["eval_loss"],
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

                # update history
                running_train_loss_hist.append(running_train_loss)
                perplexity_hist.append(metrics["perplexity"])

            # Check if reached maximum number of steps
            if completed_steps >= config.training.max_train_steps:
                break

        # end: for step
        # save checkpoint & state if needed
        if config.base.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if config.base.output_dir is not None:
                output_dir = os.path.join(config.base.output_dir, output_dir)
            accelerator.save_state(output_dir)

    # end: for epoch
    # ***** END OF TRAINING *****

    if config.base.training_memory_usage_only:
        # save model and final results
        if config.base.output_dir is not None:
            accelerator.wait_for_everyone()

            # save final results
            results = {
                "running_train_loss": running_train_loss,
                "running_train_perplexity": math.exp(running_train_loss),
                "running_train_loss_hist": running_train_loss_hist,
                **mem_usage,
            }

            os.makedirs(config.base.output_dir, exist_ok=True)
            with open(os.path.join(config.base.output_dir, "all_results.json"), "w") as f:
                json.dump(results, f, indent=4)
        exit(0)

    # disable caching
    if quant:
        model.disable_caching()
        model.clear_cache()

    # *** Evaluation (after training) ***
    s_eval.start()
    metrics = evaluate(
        config,
        model,
        eval_dataloader,
        accelerator,
        max_num_batches=config.data.validation_num_batches,
    )
    logger.info(
        f"After training -- perplexity: {metrics['perplexity']} "
        f"eval_loss: {metrics['eval_loss']}"
    )
    get_and_log_cuda_memory(logger, "After final evaluation")

    # eval on extra datasets, if needed
    if config.llm_extra.eval_extra_datasets:
        extra_metrics = _eval_extra_datasets(config, accelerator, model, tokenizer, logger)
        logger.info(f"Extra dataset results:\n{pformat(extra_metrics)}")
    s_eval.stop()

    # log metrics
    if config.logging.with_tracking:
        accelerator.log(
            {
                "running_train_loss": running_train_loss,
                "perplexity": metrics["perplexity"],
                "eval_loss": metrics["eval_loss"],
                "step": completed_steps,
            },
            step=completed_steps,
        )

    # update history
    if completed_steps > 0:
        running_train_loss_hist.append(running_train_loss)
        perplexity_hist.append(metrics["perplexity"])

    # ***** Fuse LoRA weights *****
    metrics_fused = {}
    extra_metrics_fused = {}

    if quant and config.training.param_group in TrainableParamGroup.LORA:
        # merge LoRA weights
        _merge_lora_weights(config, model)

        # Eval after fusion
        s_eval.start()

        metrics_fused = evaluate(
            config,
            model,
            eval_dataloader,
            accelerator,
            max_num_batches=config.data.validation_num_batches,
        )
        logger.info(
            f"LoRA fused: perplexity: {metrics_fused['perplexity']} "
            f"eval_loss: {metrics_fused['eval_loss']}"
        )

        if config.llm_extra.eval_extra_datasets:
            extra_metrics_fused = _eval_extra_datasets(
                config, accelerator, model, tokenizer, logger
            )
            logger.info(f"Extra dataset results (fused):\n{pformat(extra_metrics_fused)}")

        s_eval.stop()

    # ***** Apply PTQ at the end *****
    if quant and config.llm_quant.ptq_at_the_end:
        logger.info(f"Applying PTQ at the end of training")

        config.quant.weight_quant = True
        model.set_quant_state(
            weight_quant=config.quant.weight_quant, act_quant=config.quant.act_quant
        )
        model = ptq_apply_range_estimation(config, model, train_dataloader)

        # Eval after PTQ
        s_eval.start()

        metrics_fused = evaluate(
            config,
            model,
            eval_dataloader,
            accelerator,
            max_num_batches=config.data.validation_num_batches,
        )
        logger.info(
            f"After PTQ: perplexity: {metrics_fused['perplexity']} "
            f"eval_loss: {metrics_fused['eval_loss']}"
        )

        if config.llm_extra.eval_extra_datasets:
            extra_metrics_fused = _eval_extra_datasets(
                config, accelerator, model, tokenizer, logger
            )
            logger.info(f"Extra dataset results (PTQ):\n{pformat(extra_metrics_fused)}")

        s_eval.stop()

    # log training time
    train_time = s_train.get_total_duration()
    eval_time = s_eval.get_total_duration()
    logger.info(f">>> Training done")
    logger.info(f">> Running train loss: {running_train_loss:.3f}")
    logger.info(f">> Total train time:. {s_train.format()}\n")
    logger.info(f">> Total eval time:. {s_eval.format()}\n")

    if config.logging.with_tracking:
        accelerator.end_training()

    # save model and final results
    if config.base.output_dir is not None:
        accelerator.wait_for_everyone()

        # save final results
        results = {
            "running_train_loss": running_train_loss,
            "running_train_perplexity": math.exp(running_train_loss),
            "running_train_loss_hist": running_train_loss_hist,
            "perplexity": metrics["perplexity"],
            "perplexity_hist": perplexity_hist,
            "time_train": train_time,
            "time_eval": eval_time,
        }

        # add all the mem usage
        results.update(mem_usage)

        # add perplexity results
        for name, ppl in extra_metrics.items():
            results[f"perplexity_{name}"] = ppl
        if "perplexity" in metrics_fused:
            results["perplexity_fused"] = metrics_fused["perplexity"]
        for name, ppl in extra_metrics_fused.items():
            results[f"perplexity_{name}_fused"] = ppl

        os.makedirs(config.base.output_dir, exist_ok=True)
        with open(os.path.join(config.base.output_dir, "all_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # save model and tokenizer
        if config.base.save_model:
            unwrapped_model = accelerator.unwrap_model(model)
            if quant:
                torch.save(
                    unwrapped_model.state_dict(), os.path.join(config.base.output_dir, "model.pth")
                )
            else:
                unwrapped_model.save_pretrained(
                    config.base.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )

            # tokenizer
            if not quant and accelerator.is_main_process:
                tokenizer.save_pretrained(config.base.output_dir)

    # Run LM-Evaluation-Harness
    if config.lmeval.lmeval:
        logger.info("Running LM-EVAL")
        lm_eval_output = lm_eval(model, tokenizer, config.lmeval, verbose=True)

        # save results
        if config.base.output_dir is not None and lm_eval_output is not None:
            os.makedirs(config.base.output_dir, exist_ok=True)
            with open(os.path.join(config.base.output_dir, "all_lmeval_results.json"), "w") as f:
                json.dump(lm_eval_output, f, indent=4)

        # save results into all_results.json, for easier access
        if config.base.output_dir is not None and "results" in lm_eval_output:
            for task, task_results in lm_eval_output["results"].items():
                for metric, value in task_results.items():
                    name = f"{task}__{metric}"
                    results[name] = value

            # overwrite all_results.json now with extra results from LM eval
            with open(os.path.join(config.base.output_dir, "all_results.json"), "w") as f:
                json.dump(results, f, indent=4)


def _validate(config: DotDict, quant_method: Optional[BaseEnumOptions] = None):
    # setup accelerator and logger
    accelerator = init_accelerator_and_logger(config, logger)

    # log run options
    logger.info("Running with options:")
    logger.info(pformat(config))

    # set seed
    if config.base.seed is not None:
        logger.info(f"Setting the random seed to {config.base.seed}")
        set_seed(config.base.seed)

    # setup model and tokenizer
    logging.info("Loading/Setup model and tokenizer")
    model, tokenizer = make_model_and_tokenizer(config, logger)

    # load datasets
    logging.info("Loading and tokenizing dataset")
    train_dataset, eval_dataset = load_and_tokenize_datasets(config, tokenizer, accelerator, logger)

    # make dataloaders
    if quant_method in (ValidationQuantMethod.ptq, ValidationQuantMethod.mpq):
        train_dataloader = make_dataloader(config, train_dataset, train=True, multiprocessing=False)
    eval_dataloader = make_dataloader(config, eval_dataset, train=False, multiprocessing=False)

    # Quantize the model with PTQ if needed and prepare the model with accelerate
    metrics = OrderedDict()
    if quant_method == ValidationQuantMethod.ptq:
        logging.info("Running PTQ on model")
        model = ptq_main(config, model, train_dataloader, logger=logger, accelerator=accelerator)
        eval_dataloader = accelerator.prepare(eval_dataloader)

    elif quant_method == ValidationQuantMethod.no_quant:
        model = accelerator.prepare(model)
        eval_dataloader = accelerator.prepare(eval_dataloader)

    else:
        raise ValueError(f'Unknown validation quant. method "{quant_method}"')

    # Evaluate Perplexity
    if not config.base.skip_perplexity_eval:
        logging.info("Running Perplexity Evaluation")

        # run evaluation
        metrics = evaluate(
            config,
            model,
            eval_dataloader,
            accelerator,
            max_num_batches=config.data.validation_num_batches,
            metrics=metrics,
        )
        logger.info(f"perplexity: {metrics['perplexity']:.4f}")

        get_and_log_cuda_memory(logger, "After evaluation")

        # eval on extra datasets, if needed
        if config.llm_extra.eval_extra_datasets:
            extra_metrics = _eval_extra_datasets(config, accelerator, model, tokenizer, logger)
            logger.info(f"Extra dataset results:\n{pformat(extra_metrics)}")

            for name, ppl in extra_metrics.items():
                metrics[f"perplexity_{name}"] = ppl

        # save results
        if config.base.output_dir is not None:
            os.makedirs(config.base.output_dir, exist_ok=True)
            with open(os.path.join(config.base.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics, f, indent=4)

    # Run LM-Evaluation-Harness
    if "lmeval" in config.keys() and config.lmeval.lmeval:
        logger.info("Running LM-EVAL")
        lm_eval_output = lm_eval(model, tokenizer, config.lmeval, verbose=True)

        # save results
        if config.base.output_dir is not None and lm_eval_output is not None:
            os.makedirs(config.base.output_dir, exist_ok=True)
            with open(os.path.join(config.base.output_dir, "all_lmeval_results.json"), "w") as f:
                json.dump(lm_eval_output, f, indent=4)

        # save results into all_results.json, for easier access
        if config.base.output_dir is not None and "results" in lm_eval_output:
            for task, task_results in lm_eval_output["results"].items():
                for metric, value in task_results.items():
                    name = f"{task}__{metric}"
                    metrics[name] = value

            # overwrite all_results.json now with extra results from LM eval
            with open(os.path.join(config.base.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics, f, indent=4)


@click_group.command()
@pass_config
@llm_combined_base_options
@llm_training_options
@llm_extra_options
@lm_eval_options
def train_baseline(config):
    _train(config, quant=False)


@click_group.command()
@pass_config
@llm_combined_base_options
@llm_training_options
@quantization_options
@activation_quantization_options
@qat_options
@llm_quant_options
@llm_extra_options
@lm_eval_options
def train_quantized(config):
    _train(config, quant=True)


@click_group.command()
@pass_config
@llm_combined_base_options
@llm_extra_options
@lm_eval_options
def validate_baseline(config):
    _validate(config, quant_method=ValidationQuantMethod.no_quant)


@click_group.command()
@pass_config
@quantization_options
@activation_quantization_options
@llm_combined_base_options
@llm_extra_options
@lm_eval_options
def validate_quantized(config):
    _validate(config, quant_method=ValidationQuantMethod.ptq)


if __name__ == "__main__":
    click_group()
