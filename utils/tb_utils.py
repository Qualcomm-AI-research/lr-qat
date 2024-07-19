# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import warnings

from accelerate.tracking import TensorBoardTracker


def get_tb_writer(accelerator):
    for tracker in accelerator.trackers:
        if isinstance(tracker, TensorBoardTracker):
            return tracker.writer
    return None


def tb_log_scalars(config, model, accelerator, completed_steps, act_dict, logger):
    if (
        config.logging.with_tracking
        and completed_steps % config.logging.tb_scalar_log_interval == 0
    ):

        logger.info(f"Logging scalars to TensorBoard (step = {completed_steps})")

        # weight inf-norms
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                w = module.weight
                w_inf_norm = max(w.max().item(), -w.min().item())
                accelerator.log({f"{name}.weight_inf_norm": w_inf_norm}, step=completed_steps)

        # activation inf-norms
        for name, x in act_dict.items():
            x_inf_norm = max(x.max().item(), -x.min().item())
            accelerator.log({f"{name}.act_inf_norm": x_inf_norm}, step=completed_steps)


def tb_log_histograms(config, model, accelerator, completed_steps, act_dict, logger):
    if (
        config.logging.with_tracking
        and accelerator.is_main_process
        and completed_steps % config.logging.tb_hist_log_interval == 0
    ):

        logger.info(f"Logging histograms to TensorBoard (step = {completed_steps})")

        # get TB writer
        tb_writer = get_tb_writer(accelerator)
        if tb_writer is None:
            raise RuntimeError(f"Unable to retrieve TensorBoard writer")

        # weight histograms
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                w = module.weight
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)
                        tb_writer.add_histogram(
                            f"{name}.weight_hist", w, global_step=completed_steps
                        )
                except:
                    logger.warning(
                        f"Could not log weight histogram for {name} at step {completed_steps}"
                    )

        # act histograms
        for name, x in act_dict.items():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    tb_writer.add_histogram(f"{name}.act_hist", x, global_step=completed_steps)
            except:
                logger.warning(f"Could not log act histogram for {name} at step {completed_steps}")
