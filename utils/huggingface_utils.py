# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""HuggingFace utils."""
import logging
import math
import random
from collections import OrderedDict
from glob import glob
from itertools import chain
from pathlib import Path
from typing import Optional

import datasets
import torch
import transformers
import yaml
from accelerate import Accelerator
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    MixtralForCausalLM,
    default_data_collator,
)
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer

from utils import Stopwatch, convert_transformer_float_input
from utils.enums import DatasetSetup, PreprocessingType


def init_accelerator_and_logger(config, logger):
    # initialize accelerator
    accelerator_kwargs = {}
    if "training" in config:
        accelerator_kwargs["gradient_accumulation_steps"] = (
            config.training.gradient_accumulation_steps
        )

    if config.logging.with_tracking:
        accelerator_kwargs["log_with"] = config.logging.report_to
        accelerator_kwargs["project_dir"] = config.base.output_dir

    accelerator = Accelerator(**accelerator_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return accelerator


def make_model_and_tokenizer(config, logger):
    # See more about loading any type of standard or custom dataset (from files, python dict,
    # pandas DataFrame, etc) at https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process
    # can concurrently download model & vocab.
    hf_config_kwargs = {
        "cache_dir": config.model.model_cache_dir,
    }
    if config.model.config_name is not None and config.model.model_name_or_path is None:
        hf_config = AutoConfig.for_model(config.model.config_name, **hf_config_kwargs)
    elif config.model.config_name:
        hf_config = AutoConfig.from_pretrained(config.model.config_name, **hf_config_kwargs)
    elif config.model.model_name_or_path:
        hf_config = AutoConfig.from_pretrained(config.model.model_name_or_path, **hf_config_kwargs)
    else:
        hf_config = CONFIG_MAPPING[config.model.model_type]()
        logger.warning("You are instantiating a new HuggingFace config instance from scratch.")

    # Load model hf_config changes from file, if provided
    if config.model.config_path is not None:
        logger.info(f"Loading model hf_config changes from {config.model.config_path} ...")
        with open(config.model.config_path) as f:
            hf_config_changes = yaml.safe_load(f)

        for key, value in hf_config_changes.items():
            setattr(hf_config, key, value)

    # Set dropout rates, if specified
    if "training" in config:
        if config.training.attn_dropout is not None:
            logger.info(f"Setting attention dropout rate to {config.training.attn_dropout}")
            hf_config.attention_probs_dropout_prob = config.training.attn_dropout
            hf_config.attention_dropout = config.training.attn_dropout

        if config.training.hidden_dropout is not None:
            logger.info(f"Setting hidden dropout rate to {config.training.hidden_dropout}")
            hf_config.hidden_dropout_prob = config.training.hidden_dropout
            if config.model.model_type == "opt":
                hf_config.dropout = config.training.hidden_dropout

    # set keys,values caching
    logger.info(f"Setting use_cache to {config.model.use_cache}")
    hf_config.use_cache = config.model.use_cache

    # Gradient checkpointing
    logger.info(f"Setting gradient_checkpointing to {config.model.gradient_checkpointing}")
    hf_config.gradient_checkpointing = config.model.gradient_checkpointing

    # Display hf_config after changes
    logger.info("HuggingFace config after user changes:")
    logger.info(str(hf_config))

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": config.model.model_cache_dir,
    }
    tokenizer_cls = LlamaTokenizer if config.model.model_type == "llama" else AutoTokenizer
    if config.model.tokenizer_name:
        tokenizer = tokenizer_cls.from_pretrained(
            config.model.tokenizer_name,
            use_fast=not config.model.use_slow_tokenizer,
            **tokenizer_kwargs,
        )
    elif config.model.model_name_or_path:
        try:
            tokenizer = tokenizer_cls.from_pretrained(
                config.model.model_name_or_path,
                use_fast=not config.model.use_slow_tokenizer,
                legacy=False,
                **tokenizer_kwargs,
            )
        except TypeError:  # llama-3
            tokenizer = AutoTokenizer.from_pretrained(
                config.model.model_name_or_path,
                use_fast=not config.model.use_slow_tokenizer,
                legacy=False,
                **tokenizer_kwargs,
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load and prepare model
    if config.model.model_name_or_path:
        if config.model.model_type == "llama":
            model_cls = LlamaForCausalLM
        elif config.model.model_type == "mixtral":
            model_cls = MixtralForCausalLM
        else:
            model_cls = AutoModelForCausalLM

        from_pretrained_kwargs = {}
        if config.base.use_fp16:
            from_pretrained_kwargs["torch_dtype"] = torch.float16

        model = model_cls.from_pretrained(
            config.model.model_name_or_path,
            from_tf=bool(".ckpt" in config.model.model_name_or_path),
            config=hf_config,
            low_cpu_mem_usage=config.model.low_cpu_mem_usage,
            cache_dir=config.model.model_cache_dir,
            **from_pretrained_kwargs,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(hf_config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating
    # a model from scratch on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # convert the model to fp16/bf16 if required by config.
    if config.base.use_fp16:
        logger.info("Converting model to FP16 dtype")
        model = model.to(torch.float16)

    if config.base.use_bf16:
        logger.info("Converting model to BF16 dtype")
        model = model.to(torch.bfloat16)

    return model, tokenizer


def _try_load_pretokenized_datasets(config, logger, tokenized_configuration=None):
    # (model_type, data_setup, block_size, train_percentage, validation_percentage) -> tokenized_dirname
    pre_tokenized_path_map = OrderedDict(
        [
            # Mistral
            (
                ("mistral", DatasetSetup.wikitext_2, 512, None, None),
                "tokenized_wikitext_2_Mistral_512_join_nn",
            ),
            (
                ("mistral", DatasetSetup.wikitext_2, 1024, None, None),
                "tokenized_wikitext_2_Mistral_1024_join_nn",
            ),
            (
                ("mistral", DatasetSetup.wikitext_2, 2048, None, None),
                "tokenized_wikitext_2_Mistral_2048_join_nn",
            ),
        ]
    )

    # two sets of paths for Llama-{<3} and 3
    if config.llm_extra.model_id == "llama3":
        llama_path_map = OrderedDict(
            [
                (
                    ("llama", DatasetSetup.wikitext_2, 512, None, None),
                    "tokenized_wikitext_2_Llama3_512_join_nn",
                ),
                (
                    ("llama", DatasetSetup.wikitext_2, 1024, None, None),
                    "tokenized_wikitext_2_Llama3_1024_join_nn",
                ),
                (
                    ("llama", DatasetSetup.wikitext_2, 2048, None, None),
                    "tokenized_wikitext_2_Llama3_2048_join_nn",
                ),
            ]
        )
    else:  # Llama-1/2
        llama_path_map = OrderedDict(
            [
                (
                    ("llama", DatasetSetup.bookcorpus_and_wiki, 1024, None, None),
                    "tokenized_book_wiki_Llama_1024",
                ),
                (
                    ("llama", DatasetSetup.wikitext_2, 512, None, None),
                    "tokenized_wikitext_2_Llama_512_join_nn",
                ),
                (
                    ("llama", DatasetSetup.wikitext_2, 1024, None, None),
                    "tokenized_wikitext_2_Llama_1024_join_nn",
                ),
                (
                    ("llama", DatasetSetup.wikitext_2, 2048, None, None),
                    "tokenized_wikitext_2_Llama_2048_join_nn",
                ),
                (("llama", DatasetSetup.ptb, 1024, None, None), "tokenized_ptb_Llama_1024_join_nn"),
                (("llama", DatasetSetup.ptb, 2048, None, None), "tokenized_ptb_Llama_2048_join_nn"),
            ]
        )
    pre_tokenized_path_map.update(**llama_path_map)

    for k, v in pre_tokenized_path_map.items():
        pre_tokenized_path_map[k] = Path(config.data.data_cache_dir) / v

    if tokenized_configuration is None:
        tokenized_configuration = (
            config.model.model_type,
            config.data.dataset_setup,
            config.model.block_size,
            config.data.train_percentage,
            config.data.validation_percentage,
        )

    pre_tokenized_path = pre_tokenized_path_map.get(tokenized_configuration, None)

    if pre_tokenized_path is not None and pre_tokenized_path.exists():
        pre_tokenized_path = str(pre_tokenized_path)

        logger.info(f"Loading pre-tokenized dataset from {pre_tokenized_path} ...")
        tokenized_datasets = load_from_disk(pre_tokenized_path)
        return tokenized_datasets
    elif pre_tokenized_path is not None:
        logger.info(f"Pre-tokenized dataset not found in {pre_tokenized_path}.")
    else:
        logger.info(f"Pre-tokenized dataset config do not exist.")

    return None


def _load_and_tokenize_raw_datasets(config, tokenizer, accelerator, logger):
    # In distributed training, the load_dataset function guarantee that only one local process can
    # concurrently download the dataset.
    train_percentage = config.data.train_percentage
    val_percentage = config.data.validation_percentage

    train_split = "train" if train_percentage is None else f"train[:{train_percentage}%]"
    val_split = "validation" if val_percentage is None else f"validation[:{val_percentage}%]"

    # Loading or downloading the dataset
    logger.info(
        f"Loading or downloading `{config.data.dataset_setup.name}` dataset from from HuggingFace..."
    )

    if config.data.dataset_setup == DatasetSetup.wikitext_2:
        raw_datasets = DatasetDict()
        raw_datasets["train"] = load_dataset(
            "wikitext", "wikitext-2-raw-v1", cache_dir=config.data.data_cache_dir, split=train_split
        )
        raw_datasets["validation"] = load_dataset(
            "wikitext", "wikitext-2-raw-v1", cache_dir=config.data.data_cache_dir, split=val_split
        )
        # for Wikitext-2, we also use test set
        raw_datasets["test"] = load_dataset(
            "wikitext", "wikitext-2-raw-v1", cache_dir=config.data.data_cache_dir, split="test"
        )

    elif config.data.dataset_setup == DatasetSetup.wikitext_103:
        raw_datasets = DatasetDict()
        raw_datasets["train"] = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            cache_dir=config.data.data_cache_dir,
            split=train_split,
        )
        raw_datasets["validation"] = load_dataset(
            "wikitext", "wikitext-103-raw-v1", cache_dir=config.data.data_cache_dir, split=val_split
        )

    elif config.data.dataset_setup == DatasetSetup.ptb:
        raw_datasets = DatasetDict()
        raw_datasets["train"] = load_dataset(
            "ptb_text_only", cache_dir=config.data.data_cache_dir, split=train_split
        )
        raw_datasets["validation"] = load_dataset(
            "ptb_text_only", cache_dir=config.data.data_cache_dir, split=val_split
        )

    elif config.data.dataset_setup == DatasetSetup.bookcorpus_and_wiki:
        bookcorpus = load_dataset(
            "bookcorpus", cache_dir=config.data.data_cache_dir, split=train_split
        )

        wiki_train = load_dataset(
            "wiki40b", "en", cache_dir=config.data.data_cache_dir, split=train_split
        )
        wiki_val = load_dataset(
            "wiki40b", "en", cache_dir=config.data.data_cache_dir, split=val_split
        )

        # only keep the 'text' column
        wiki_train = wiki_train.remove_columns([c for c in wiki_train.column_names if c != "text"])
        wiki_val = wiki_val.remove_columns([col for col in wiki_val.column_names if col != "text"])
        assert bookcorpus.features.type == wiki_train.features.type

        raw_datasets = DatasetDict()
        raw_datasets["train_book"] = bookcorpus
        raw_datasets["train_wiki"] = wiki_train
        raw_datasets["validation"] = wiki_val

    else:
        raise ValueError(f"Unknown dataset, {config.data.dataset_setup}")

    # Preprocessing the datasets.
    logger.info(f"Pre-processing the dataset `{config.data.dataset_setup.name}` just downloaded...")

    # Check sequence length
    if config.model.block_size is None:
        logger.warning(f"block size is not specified, setting it to the model max seq length")

        config.model.block_size = tokenizer.model_max_length
        if config.model.block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the "
                "default `block_size` value of 1024. If you would like to use a longer "
                "`block_size` up to `tokenizer.model_max_length` you can override this default "
                "with `--block_size xxx`."
            )
        config.model.block_size = 1024

    if config.model.block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({config.model.block_size}) is larger than the maximum "
            f"length for the model ({tokenizer.model_max_length}). Using "
            f"block_size={tokenizer.model_max_length}."
        )
    block_size = min(config.model.block_size, tokenizer.model_max_length)

    # Tokenize all the texts.
    column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Define tokenization function
    def tokenize_function_line_by_line(examples):
        return tokenizer(examples[text_column_name])

    def tokenize_function_join(examples):
        return tokenizer(["".join(examples[text_column_name])])

    def tokenize_function_join_nn(examples):
        return tokenizer(["\n\n".join(examples[text_column_name])])

    if config.data.preprocessing_type == PreprocessingType.line_by_line:
        tokenize_function = tokenize_function_line_by_line
    elif config.data.preprocessing_type == PreprocessingType.join:
        tokenize_function = tokenize_function_join
    elif config.data.preprocessing_type == PreprocessingType.join_nn:
        tokenize_function = tokenize_function_join_nn

    # make the default batch size for text pre-processing explicit
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            batch_size=config.data.tokenizer_batch_size,
            writer_batch_size=config.data.tokenizer_batch_size,
            num_proc=config.data.num_workers if config.data.num_workers > 0 else 1,
            remove_columns=column_names,
            load_from_cache_file=not config.data.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Main data processing function that will concatenate all texts from our dataset and generate
    # chunks of max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of
        # this drop, you can customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            total_length = 0
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts
    # throws away a remainder for each of those groups of 1,000 texts. You can adjust that
    # batch_size here but a higher value might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method
    # for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=config.data.tokenizer_batch_size,
            num_proc=config.data.num_workers if config.data.num_workers > 0 else 1,
            load_from_cache_file=not config.data.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return tokenized_datasets


def _load_slimpajama_dataset(logger, config):
    s_slimpajama = Stopwatch()
    s_slimpajama.start()

    slimpajama_path = config.data.config.data.slimpajama_pretokenized_dir
    logger.info(f"Loading pre-tokenized SlimPajama from {slimpajama_path}")
    if "*" in slimpajama_path:
        dataset = concatenate_datasets(
            [Dataset.from_file(fpath) for fpath in list(sorted(glob(slimpajama_path)))]
        )
    else:
        dataset = Dataset.from_file(slimpajama_path)

    logger.info(dataset)
    logger.info(len(dataset[0]["input_ids"]))
    logger.info(dataset[0]["input_ids"][:10])
    logger.info(f"Loaded Slimpajama dataset successfully: {s_slimpajama.stop().format()}")
    return dataset


def load_and_tokenize_datasets(config, tokenizer, accelerator, logger):
    # SlimPajama + Wiki
    if config.data.dataset_setup == DatasetSetup.slimpajama_wiki:
        if config.model.model_type != "llama":
            raise ValueError(f"Dataset setup `slimpajama_wiki` is only supported for Llama")

        # load wiki first
        tokenized_configuration = (
            config.model.model_type,
            DatasetSetup.bookcorpus_and_wiki,
            config.model.block_size,
            config.data.train_percentage,
            config.data.validation_percentage,
        )
        datasets = _try_load_pretokenized_datasets(config, logger, tokenized_configuration)
        if datasets is None:
            raise RuntimeError(f"No pre-tokenized wikipedia is available for `slimpajama_wiki`.")
        eval_dataset = datasets["validation"]
        logger.info(f"Loaded Wikipedia validation set successfully")

        # load slimpajama
        train_dataset = _load_slimpajama_dataset(logger, config)

        return train_dataset, eval_dataset

    # SlimPajama + Wikitext-2
    if config.data.dataset_setup == DatasetSetup.slimpajama_wikitext_2:
        if not config.model.model_type in ("llama", "mistral"):
            raise ValueError(
                f"Dataset setup `slimpajama_wikitext_2` is only supported for Llama "
                f"and Mistral models"
            )

        # load wikitext-2 first
        tokenized_configuration = (
            config.model.model_type,
            DatasetSetup.wikitext_2,
            config.model.block_size,
            config.data.train_percentage,
            config.data.validation_percentage,
        )

        datasets = _try_load_pretokenized_datasets(config, logger, tokenized_configuration)
        if datasets is None:
            raise RuntimeError(
                f"No pre-tokenized wikitext-2 is available for " f"`slimpajama_wikitext_2`."
            )
        eval_dataset = datasets["test"]  # use test set for validation on Wikitext-2
        logger.info(f"Loaded Wikitext-2 TEST set successfully")

        # load slimpajama
        train_dataset = _load_slimpajama_dataset(logger, config)
        return train_dataset, eval_dataset

    # Get tokenized datasets.
    # first check if we have pre-tokenized datasets stored and if so, load them directly:
    tokenized_datasets = _try_load_pretokenized_datasets(config, logger)

    if tokenized_datasets is None:
        # otherwise, load using HuggingFace + do tokenization
        tokenized_datasets = _load_and_tokenize_raw_datasets(config, tokenizer, accelerator, logger)

    # Split into train/val
    if config.data.dataset_setup == DatasetSetup.wikitext_2:
        # use test set for wikitext-2, to compare with the literature
        eval_dataset = tokenized_datasets["test"]
    else:
        eval_dataset = tokenized_datasets["validation"]

    if config.data.dataset_setup == DatasetSetup.bookcorpus_and_wiki:
        train_dataset = concatenate_datasets(
            [tokenized_datasets["train_book"], tokenized_datasets["train_wiki"]]
        )
    else:
        train_dataset = tokenized_datasets["train"]

    # Log a few random samples from the training set:
    if len(train_dataset) > 3:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    return train_dataset, eval_dataset


def make_dataloader(config, dataset, train=True, multiprocessing=True):
    if train:
        bs = config.base.per_device_train_batch_size
        shuffle = True
    else:
        bs = config.base.per_device_eval_batch_size
        shuffle = False

    num_workers = config.data.num_workers if multiprocessing else 0
    if num_workers > 0:
        logging.info(f"Using multiprocessing for dataloader: num_workers={num_workers}")
    else:
        logging.info(f"Disabling multiprocessing for dataloader: num_workers={num_workers}")

    dataloader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


def evaluate(
    config,
    model,
    eval_dataloader,
    accelerator,
    max_num_batches=None,
    metrics: Optional[OrderedDict] = None,
):

    # *** Evaluation ***
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(tqdm(eval_dataloader)):

        if max_num_batches is not None and batch_idx >= max_num_batches:
            break

        batch = convert_transformer_float_input(
            batch, bf16=config.base.use_bf16, fp16=config.base.use_fp16
        )

        with torch.no_grad():
            outputs = model(**batch)

        try:
            loss = outputs.loss
        except AttributeError:
            if isinstance(outputs, tuple):
                loss = outputs[0]
                assert len(loss.shape) == 0  # we are looking for a scalar

        loss_ = accelerator.gather_for_metrics(loss.repeat(config.base.per_device_eval_batch_size))
        losses.append(loss_)

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    # metrics
    metrics_ = OrderedDict([("eval_loss", eval_loss), ("perplexity", perplexity)])
    if metrics is not None:
        metrics.update(metrics_)
        return metrics
    return metrics_
