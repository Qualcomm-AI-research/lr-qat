# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import json

from utils import DotDict


def lm_eval(model, tokenizer, config: DotDict, verbose=False):
    """Run lm-eval given a model, a tokenizer, and a config dict."""

    def vprint(str):
        if verbose:
            print(str)

    if config.lmeval is True:
        vprint("Running LM-Eval")

        from lm_eval import evaluator as lmeval_evaluator
        from lm_eval.models.huggingface import HFLM

        lm_eval_model = HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=config.lmeval_batch_size
        )

        results = lmeval_evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=list(config.lmeval_task),
            num_fewshot=config.lmeval_num_fewshot,
            batch_size=config.lmeval_batch_size,
            limit=config.lmeval_limit,
        )

        if not isinstance(results["config"]["model"], str):
            results["config"]["model"] = str(model.__class__)

        # way too much data (all samples are printed)
        if "samples" in results:
            del results["samples"]

        vprint(json.dumps(results, indent=4))
        return results

    else:
        vprint("LM-Eval skipped.")
        return None
