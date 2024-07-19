# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
from functools import wraps

import click

from lmeval.tasks import DEFAULT_TASKS
from utils.click_utils import split_dict


def lm_eval_options(func):
    @click.option("--lmeval", is_flag=True, default=False, help="Enable the lm-evaluation harness.")
    @click.option(
        "--lmeval-task",
        default=DEFAULT_TASKS,
        multiple=True,
        type=click.Choice(DEFAULT_TASKS),
        help="LM-Eval tasks to be executed, you can select multiple tasks by repeating the command.",
    )
    @click.option("--lmeval-num-fewshot", type=int, default=0)
    @click.option("--lmeval-batch-size", type=int, default=1)
    @click.option("--lmeval-output_path", type=str, default=None)
    @click.option("--lmeval-limit", type=int, default=None)
    @click.option("--lmeval-no-cache", is_flag=True, default=False)
    @click.option("--lmeval-decontamination-ngrams-path", type=str, default=None)
    @click.option("--lmeval-description-dict-path", type=str, default=None)
    @click.option("--lmeval-check-integrity", is_flag=True, default=False)
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        attrs = [
            "lmeval",
            "lmeval_task",
            "lmeval_num_fewshot",
            "lmeval_batch_size",
            "lmeval_output_path",
            "lmeval_limit",
            "lmeval_no_cache",
            "lmeval_decontamination_ngrams_path",
            "lmeval_description_dict_path",
            "lmeval_check_integrity",
        ]
        config.lmeval, other_kw = split_dict(kwargs, attrs)

        return func(config, *args, **other_kw)

    return func_wrapper
