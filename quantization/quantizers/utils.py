# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import inspect
from typing import Iterable

import torch
import torch.nn as nn


class QuantizerNotInitializedError(Exception):
    """Raised when a quantizer has not been initialized"""

    def __init__(self):
        super(QuantizerNotInitializedError, self).__init__(
            "Quantizer has  not been initialized yet"
        )


def parameter_to_buffer(module: nn.Module, name: str, strict: bool = False):
    """
    Convert the underlying Tensor of a nn.Parameter object to a buffer. This is
    currently used to 'fix' learnable ranges.

    NB: this also requires the parameter to be removed from the optimizer, as this
    will hold refererences to the  original Parameter object.

    Args:
        module (torch.nn.Module): module to change parameter to buffer on
        name (str): name of parameter attribute to change to buffer
        strict(bool): if True, raise error if `module.{name}` is not a parameter. Otherwise,
            return without making a change, i.e., fail silently.
    """
    param = getattr(module, name)
    param_registered = any(param is reg_param for reg_param in module.parameters())
    if not (isinstance(param, nn.Parameter) and param_registered):
        if not strict:
            return
        raise ValueError(
            f"Expected 'module.{name} to be a parameter, but found {type(param).__name__}"
        )
    delattr(module, name)
    if isinstance(param, nn.parameter.UninitializedParameter):
        # pylint: disable=line-too-long
        module.register_buffer(name, nn.parameter.UninitializedBuffer(device=param.device))  # type: ignore[call-arg]
    else:
        module.register_buffer(name, param.data)


def buffer_to_parameter(module: nn.Module, name: str, strict: bool = True):
    """
    Convert tensor on a module to a nn.Parameter. This is currently used to
    'enable' learnable ranges.

    NB: this also requires the parameter to be added to the optimizer, as this
    will not have been included in an earlier `module.Parameters()` call.

    Args:
        module (torch.nn.Module): module to change buffer to parameter on
        name (str): name of buffer attribute to change to parameter
        strict(bool): if True, raise error if `module.{name}` is not a Tensor. Otherwise,
            return without making a change, i.e., fail silently.
    """
    buf = getattr(module, name)
    buf_registered = any(buf is reg_buf for reg_buf in module.buffers())
    if not (isinstance(buf, torch.Tensor) and buf_registered):
        if not strict:
            return
        raise ValueError(f"Expected 'module.{name} to be a buffer, but found {type(buf).__name__}")
    delattr(module, name)
    if isinstance(buf, nn.parameter.UninitializedBuffer):
        # pylint: disable=line-too-long
        module.register_parameter(name, nn.parameter.UninitializedParameter(device=buf.device))  # type: ignore[call-arg]
    else:
        module.register_parameter(name, nn.Parameter(buf))


def create_discretizer(gradient_estimator, grad_estimator_params=None):
    if inspect.isclass(gradient_estimator):
        # The discretizer is an nn.Module
        if grad_estimator_params is not None:
            if isinstance(grad_estimator_params, Iterable):
                # The gradient estimator parameters are iterable
                discretizer = gradient_estimator(*grad_estimator_params)
            else:
                # The gradient estimator parameter is a constant
                discretizer = gradient_estimator(grad_estimator_params)
        else:
            # No parameter required
            discretizer = gradient_estimator()
    else:
        # The discretizer is a functional
        discretizer = gradient_estimator

    return discretizer
