# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import sys
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch


class DotDict(dict):
    """
    This class enables access to its attributes as both ['attr'] and .attr .
    Its advantage is that content of its `instance` can be accessed with `.`
    and still passed to functions as `**instance` (as dictionaries) for
    implementing variable-length arguments.
    """

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")


class Stopwatch:
    """
    A simple cross-platform context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as st:
    ...     time.sleep(0.101)  #doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    """

    def __init__(self, name=None, verbose=False):
        self._name = name
        self._verbose = verbose

        self._start_time_point = 0.0
        self._total_duration = 0.0
        self._is_running = False

        if sys.platform == "win32":
            # on Windows, the best timer is time.clock()
            self._timer_fn = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self._timer_fn = time.time

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._verbose:
            self.print()

    def start(self):
        if not self._is_running:
            self._start_time_point = self._timer_fn()
            self._is_running = True
        return self

    def stop(self):
        if self._is_running:
            self._total_duration += self._timer_fn() - self._start_time_point
            self._is_running = False
        return self

    def reset(self):
        self._start_time_point = 0.0
        self._total_duration = 0.0
        self._is_running = False
        return self

    def _update_state(self):
        now = self._timer_fn()
        self._total_duration += now - self._start_time_point
        self._start_time_point = now

    def _format(self):
        prefix = f"[{self._name}]" if self._name is not None else "Elapsed time"
        info = f"{prefix}: {self._total_duration:.3f} sec"
        return info

    def format(self):
        if self._is_running:
            self._update_state()
        return self._format()

    def print(self):
        print(self.format())

    def get_total_duration(self):
        if self._is_running:
            self._update_state()
        return self._total_duration


def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)


def attach_act_hooks(model):
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            act_dict[name] = out[0]

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


def apply_transformer_float_input_dtype(batch: dict, dtype: torch.dtype) -> dict:
    new_batch = {}
    for k, v in batch.items():
        if v.dtype == torch.float32:
            new_batch[k] = v.to(dtype)
        else:
            new_batch[k] = v
    return new_batch


def convert_transformer_float_input(
    batch: dict, bf16: Optional[bool] = False, fp16: Optional[bool] = False
) -> dict:
    if bf16:
        batch = apply_transformer_float_input_dtype(batch, dtype=torch.bfloat16)
    elif fp16:
        batch = apply_transformer_float_input_dtype(batch, dtype=torch.float16)
    return batch


def truncate_batch_to_block_size(batch: dict, block_size: int) -> dict:
    new_batch = {}
    for k, v in batch.items():
        new_batch[k] = v[:, :block_size]
    return new_batch


def get_and_log_cuda_memory(logger, prefix=None):
    """
    log and return the following (as a dict):
    * current GPU memory occupied by tensors in bytes for a given device.
    * current GPU memory managed by the caching allocator in bytes for a given device.
    * peak cached memory since the beginning of the program
    """
    out = {}

    if prefix:
        logger.info(f"CUDA memory usage [{prefix}]")

    m_alloc = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
    m_reserved = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
    m_max_reserved = torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024

    logger.info(f"torch.cuda.memory_allocated: {m_alloc:.3f} GB")
    logger.info(f"torch.cuda.memory_reserved: {m_reserved:.3f} GB")
    logger.info(f"torch.cuda.max_memory_reserved: {m_max_reserved:.3f} GB")

    out[f"cuda__{prefix}__m_alloc"] = m_alloc
    out[f"cuda__{prefix}__m_reserved"] = m_reserved
    out[f"cuda__{prefix}__m_max_reserved"] = m_max_reserved
    return out
