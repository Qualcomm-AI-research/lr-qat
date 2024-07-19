# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
from functools import partial, wraps

import click
from transformers import MODEL_MAPPING, SchedulerType

from utils import DotDict
from utils.click_utils import ClickEnumOption, split_dict
from utils.enums import (
    ActQuantBits,
    DatasetSetup,
    LR_QAT_Method,
    PreprocessingType,
    TrainableParamGroup,
)

# show default values for all options
click.option = partial(click.option, show_default=True)


# define the available model types
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def llm_base_options(func):
    @click.option("--seed", type=int, default=1000, help="Random number generator seed to set.")
    @click.option(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    @click.option(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    # dtype
    @click.option(
        "--use-fp16", is_flag=True, default=False, help="Cast model parameters and input to FP16."
    )
    @click.option(
        "--use-bf16", is_flag=True, default=False, help="Cast model parameters and input to BF16."
    )

    # saving/loading
    @click.option("--output-dir", type=str, default=None, help="Where to store the final model.")
    @click.option(
        "--checkpointing-steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or "
        '"epoch" for each epoch.',
    )
    @click.option(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    @click.option(
        "--save-model", is_flag=True, default=True, help="Save model and tokenizer after training."
    )
    @click.option(
        "--skip-perplexity-eval",
        is_flag=True,
        default=False,
        help="Skip perplexity evaluation during evaluation of the network.",
    )
    @click.option(
        "--training-memory-usage-only",
        is_flag=True,
        default=False,
        help="Just benchmark memory usage during training and quit immediately.",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "seed",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "use_fp16",
            "use_bf16",
            "output_dir",
            "checkpointing_steps",
            "resume_from_checkpoint",
            "save_model",
            "skip_perplexity_eval",
            "training_memory_usage_only",
        ]
        config.base, other_kw = split_dict(kwargs, attrs)

        return func(config, *args, **other_kw)

    return func_wrapper


def llm_data_options(func):
    @click.option(
        "--dataset-setup",
        default=DatasetSetup.wikitext_103.name,
        type=ClickEnumOption(DatasetSetup, case_sensitive=False),
        help=f"The setup/preset of the datasets to use for training/validation.",
    )
    @click.option(
        "--train-percentage", type=int, default=None, help="Percentage of training set to use."
    )
    @click.option(
        "--validation-percentage",
        type=int,
        default=None,
        help="Percentage of validation set to use.",
    )
    @click.option(
        "--validation-num-batches",
        type=int,
        default=None,
        help="If specified, run validation only on first N batches.",
    )
    @click.option(
        "--num-workers",
        type=int,
        default=0,
        help="The number of processes to use for the preprocessing.",
    )
    @click.option(
        "--preprocessing-type",
        default=PreprocessingType.join_nn.name,
        type=ClickEnumOption(PreprocessingType, case_sensitive=False),
        help=f"The type of text data preprocessing.",
    )
    @click.option(
        "--tokenizer-batch-size",
        default=None,
        type=int,
        help=f"Number of examples per batch provided to tokenizer. `batch_size <= 0` or "
        f"`batch_size == None`: Provide the full dataset as a single batch to the \
                       tokenizer.",
    )
    @click.option(
        "--data-cache-dir",
        default="~/.hf_data",
        type=click.Path(file_okay=False, writable=True, resolve_path=True),
        help="A directory where both raw and preprocessed datasets are stored.",
    )
    @click.option(
        "--slimpajama-pretokenized-path",
        default=None,
        type=str,
        help="A filepath / regexp with the pre-tokenized SlimPajama (*.arrow) dataset.",
    )
    @click.option(
        "--overwrite-cache/--no-overwrite-cache",
        is_flag=True,
        default=True,
        help="Overwrite the cached training and evaluation sets.",
    )
    @click.option(
        "--eval-on-testset",
        is_flag=True,
        type=bool,
        default=False,
        help="Use testset instead of validation set (if available).",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "dataset_setup",
            "train_percentage",
            "validation_percentage",
            "validation_num_batches",
            "num_workers",
            "preprocessing_type",
            "tokenizer_batch_size",
            "data_cache_dir",
            "slimpajama_pretokenized_path",
            "overwrite_cache",
            "eval_on_testset",
        ]
        config.data, other_kw = split_dict(kwargs, attrs)

        if (
            config.data.dataset_setup in DatasetSetup.SLIMPAJAMA
            and config.data.slimpajama_pretokenized_path is None
        ):
            raise ValueError(
                f"--slimpajama-pretokenized-path is required with dataset setup "
                f"{config.data.dataset_setup}"
            )

        return func(config, *args, **other_kw)

    return func_wrapper


def llm_model_and_tokenizer_options(func):
    @click.option(
        "--model-type",
        default=None,
        type=click.Choice(MODEL_TYPES, case_sensitive=False),
        help="Model type to use if training from scratch.",
    )
    @click.option(
        "--model-name-or-path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    @click.option(
        "--model-state-dict-path",
        type=str,
        default=None,
        help="Path to model state dict (not saved together with config or tokenizer).",
    )
    @click.option(
        "--config-name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    @click.option(
        "--config-path",
        type=str,
        default=None,
        help="Path to a yaml file with model config modifications.",
    )
    @click.option(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    @click.option(
        "--use-slow-tokenizer",
        is_flag=True,
        default=False,
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers " "library).",
    )
    @click.option(
        "--pad-to-max-length/--no-pad-to-max-length",
        is_flag=True,
        default=True,
        help="If True, pad all samples to `max_length`. Otherwise, use dynamic padding.",
    )
    @click.option(
        "--max-seq-length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences "
        "longer than this will be truncated.",
    )
    @click.option(
        "--block-size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset "
        "will be truncated in block of this size for training. Default to the model "
        "max input length for single sentence inputs (take into account secial "
        "tokens).",
    )
    @click.option(
        "--low-cpu-mem-usage",
        is_flag=True,
        default=False,
        help="It is an option to create the model as an empty shell, then only "
        "materialize its parameters when the pretrained weights are loaded. If "
        "passed, LLM loading time and RAM consumption will be benefited.",
    )
    @click.option(
        "--use-cache/--no-use-cache",
        is_flag=True,
        default=False,
        help="If use_cache is True, past key values are used to speed up decoding if "
        "applicable to model (consumes more VRAM).",
    )
    @click.option(
        "--gradient-checkpointing/--no-gradient-checkpointing",
        is_flag=True,
        default=False,
        help="Whether or not to use gradient checkpointing",
    )
    @click.option(
        "--model-cache-dir",
        default="~/.hf_cache",
        type=click.Path(file_okay=False, writable=True, resolve_path=True),
        help="Where to store downloaded pretrained HuggingFace models (together with "
        "respective config and a tokenizer).",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "model_type",
            "model_name_or_path",
            "model_state_dict_path",
            "config_name",
            "config_path",
            "tokenizer_name",
            "use_slow_tokenizer",
            "pad_to_max_length",
            "max_seq_length",
            "block_size",
            "low_cpu_mem_usage",
            "use_cache",
            "gradient_checkpointing",
            "model_cache_dir",
        ]
        config.model, other_kw = split_dict(kwargs, attrs)

        return func(config, *args, **other_kw)

    return func_wrapper


def llm_training_options(func):
    # Optimization
    @click.option(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial/maximum learning rate (after the potential warmup period) to use.",
    )
    @click.option(
        "--lr-scheduler-type",
        type=SchedulerType,
        default=SchedulerType.LINEAR.value,
        help="The scheduler type to use. Choose between ['linear', 'cosine', "
        "'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup']",
    )
    @click.option(
        "--final-lr-fraction",
        type=float,
        default=0.0,
        help="Final LR as a fraction of the maximum LR (supports only linear schedule).",
    )
    @click.option("--scales-lr", type=float, default=1e-5, help="LR for scales.")
    @click.option("--lr-ab", type=float, default=5e-5, help="A separate LR for LoRA adapters.")
    @click.option("--beta1", type=float, default=0.9, help="Adam beta1.")
    @click.option("--beta2", type=float, default=0.999, help="Adam beta1.")
    @click.option(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    @click.option(
        "--max-train-steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides "
        "num_train_epochs.",
    )
    @click.option(
        "--num-warmup-steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    @click.option(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a " "backward/update pass.",
    )
    @click.option(
        "--param-group",
        default=TrainableParamGroup.all.name,
        type=ClickEnumOption(TrainableParamGroup, case_sensitive=False),
        help=f"The set of model parameters used for training.",
    )

    # Regularization
    @click.option("--weight-decay", type=float, default=0.0, help="Weight decay to use.")
    @click.option(
        "--wd-ln-gamma",
        is_flag=True,
        default=False,
        help="If True, apply weight decay to LayerNorm gamma.",
    )
    @click.option(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Max gradient norm. If set to 0, no clipping will be applied.",
    )
    @click.option(
        "--grad-norm-type", type=float, default=2.0, help="Norm type to use for gradient clipping."
    )
    @click.option(
        "--attn-dropout", type=float, default=None, help="Dropout rate to set for attention probs."
    )
    @click.option(
        "--hidden-dropout", type=float, default=None, help="Dropout rate to set for hidden states."
    )
    # Misc
    @click.option(
        "--cpu-opt-state",
        is_flag=True,
        default=False,
        help="reduce VRAM usage by keeping the optimizer state on CPU.",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "learning_rate",
            "lr_scheduler_type",
            "final_lr_fraction",
            "scales_lr",
            "lr_ab",
            "beta1",
            "beta2",
            "num_train_epochs",
            "max_train_steps",
            "num_warmup_steps",
            "gradient_accumulation_steps",
            "param_group",
            "weight_decay",
            "wd_ln_gamma",
            "max_grad_norm",
            "grad_norm_type",
            "attn_dropout",
            "hidden_dropout",
            "cpu_opt_state",
        ]
        config.training, other_kw = split_dict(kwargs, attrs)

        # make LLM click options compatible with QAT click options:
        config.optimizer = DotDict(max_epochs=config.training.num_train_epochs)

        return func(config, *args, **other_kw)

    return func_wrapper


def llm_logging_options(func):
    @click.option(
        "--with-tracking/--no-tracking",
        is_flag=True,
        default=True,
        help="Whether to enable experiment trackers for logging.",
    )
    @click.option(
        "--report-to",
        type=str,
        default="tensorboard",
        help="The integration to report the results and logs to. Supported platforms are "
        '`"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` to '
        "report to all integrations. Only applicable when --with-tracking is "
        "passed.",
    )
    @click.option(
        "--tqdm-update-interval",
        type=int,
        default=1,
        help="How often to update tqdm progress bar. Note that setting this value to a "
        "small number will result in excessively large log files.",
    )
    @click.option(
        "--tb-detailed-logging/--no-tb-detailed-logging",
        is_flag=True,
        default=False,
        help="Whether to log weights and activations stats and histograms in TensorBoard."
        "NOTE: will increase VRAM usage!",
    )
    @click.option(
        "--tb-scalar-log-interval",
        type=int,
        default=1000,
        help="How often to log scalar stats of weights and activations to TensorBoard "
        "(in steps).",
    )
    @click.option(
        "--tb-hist-log-interval",
        type=int,
        default=10000,
        help="How often to log histograms of weights and activations to TensorBoard " "(in steps).",
    )
    @click.option(
        "--running-train-loss-gamma",
        type=float,
        default=0.99,
        help="Momentum for training loss exponential moving average.",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "with_tracking",
            "report_to",
            "tqdm_update_interval",
            "tb_detailed_logging",
            "tb_scalar_log_interval",
            "tb_hist_log_interval",
            "running_train_loss_gamma",
        ]
        config.logging, other_kw = split_dict(kwargs, attrs)

        return func(config, *args, **other_kw)

    return func_wrapper


def llm_quant_options(func):
    @click.option(
        "--act-quant-bits",
        default=ActQuantBits.a8.name,
        type=ClickEnumOption(ActQuantBits, case_sensitive=False),
        help=f"The set of model parameters used for training.",
    )
    @click.option(
        "--ptq-at-the-end",
        is_flag=True,
        default=False,
        help="Apply PTQ after QAT (assuming no weight quant and no act quant).",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "act_quant_bits",
            "ptq_at_the_end",
        ]
        config.llm_quant, other_kw = split_dict(kwargs, attrs)

        # set the bits and act pattern depending on the preset
        if config.quant.act_quant:
            if config.llm_quant.act_quant_bits == ActQuantBits.a8:
                config.quant.n_bits_act = 8
                config.act_quant.quant_pattern = "na.k.v.qe.ke.ao.nm.mm"
                # ^ a superset of quantizers from OmniQuant and LLM-QAT

            elif config.llm_quant.act_quant_bits == ActQuantBits.a8kv4:
                config.quant.n_bits_act = 8
                config.act_quant.quant_pattern = "na.k=4.v=4.qe.ke.ao.nm.mm"

            elif config.llm_quant.act_quant_bits == ActQuantBits.a4:
                config.quant.n_bits_act = 4
                config.act_quant.quant_pattern = "na.k.v.qe.ke.ao.nm.mm"

        # checks
        if config.llm_quant.ptq_at_the_end:
            if config.quant.weight_quant:
                raise ValueError(f"--ptq-at-the-end is incompatible with --weight-quant")
            if config.quant.act_quant:
                raise ValueError(f"--ptq-at-the-end is incompatible with --act-quant")

        return func(config, *args, **other_kw)

    return func_wrapper


def llm_extra_options(func):
    ## Model Id
    @click.option(
        "--model-id",
        type=str,
        default=None,
        help='Extra flag to discriminate between llama2 and llama3. For llama3 - use `--model-id "llama3"`',
    )

    ## Extra eval options
    @click.option(
        "--eval-extra-datasets/--no-eval-extra-datasets",
        is_flag=True,
        default=True,
        help="Eval on extra datasets.",
    )
    @click.option(
        "--eval-before-training/--no-eval-before-training",
        is_flag=True,
        default=False,
        help="Whether to run evaluation before training.",
    )
    @click.option("--eval-every-step", type=int, default=None, help="Eval model every N steps.")

    ## LR-QAT
    @click.option("--lora-r", type=int, default=1, help="Rank for LoRA.")
    @click.option("--lora-alpha", type=int, default=1, help="LoRA alpha.")
    @click.option("--lora-dropout", type=float, default=0.0, help="Dropout for LoRA.")
    @click.option(
        "--lora-method",
        default=LR_QAT_Method.naive.name,
        type=ClickEnumOption(LR_QAT_Method, case_sensitive=False),
        help="Method for combining LoRA with quantization.",
    )
    @click.option("--lora-b-std", type=float, default=0.0, help="Std for LoRA B init.")
    @click.option(
        "--lora-save-int-tensors",
        is_flag=True,
        default=False,
        help="Whether to save W_int and C_int tensors for Lora STE method.",
    )
    @click.option(
        "--use-checkpointing",
        is_flag=True,
        default=False,
        help="Whether to use checkpointing for LoRA + STE method.",
    )
    @click.option(
        "--use-lora-svd-init",
        is_flag=True,
        default=False,
        help="Whether to use SVD initialization with alternating optimization for init LoRA A-B weights.",
    )
    @click.option(
        "--lora-svd-init-steps",
        is_flag=False,
        default=1,
        help="How many alternating optimization steps to lora svd initialization.",
    )
    @click.option(
        "--module-quant-filter",
        type=str,
        is_flag=False,
        default=None,
        help="Select the path to the module to be quantized.",
    )
    @click.option(
        "--use-bf16-ab",
        type=bool,
        is_flag=True,
        default=False,
        help="Storea A and B matrix in BF16 format.",
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "model_id",
            "eval_extra_datasets",
            "eval_before_training",
            "eval_every_step",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_method",
            "lora_b_std",
            "lora_save_int_tensors",
            "use_checkpointing",
            "use_lora_svd_init",
            "lora_svd_init_steps",
            "module_quant_filter",
            "use_bf16_ab",
        ]
        config.llm_extra, other_kw = split_dict(kwargs, attrs)

        return func(config, *args, **other_kw)

    return func_wrapper


def llm_combined_base_options(func):
    @llm_base_options
    @llm_data_options
    @llm_model_and_tokenizer_options
    @llm_logging_options
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        return func(config, *args, **kwargs)

    return func_wrapper
