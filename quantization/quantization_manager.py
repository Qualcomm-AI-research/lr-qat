# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import warnings
from enum import auto
from typing import Optional

import torch
from torch import nn

from quantization.quantizers import QMethods
from quantization.quantizers.base_quantizers import QuantizerBase
from quantization.quantizers.utils import QuantizerNotInitializedError
from quantization.range_estimators import RangeEstimatorBase, RangeEstimators
from utils.enums import BaseEnumOptions


class Qstates(BaseEnumOptions):
    estimate_ranges = auto()  # ranges are updated in eval and train mode
    fix_ranges = auto()  # quantization ranges are fixed for train and eval
    learn_ranges = auto()  # quantization params are nn.Parameters
    estimate_ranges_train = (
        auto()
    )  # quantization ranges are updated during train and fixed for eval


class QuantizationManager(nn.Module):
    """Implementation of Quantization and Quantization Range Estimation

    Parameters
    ----------
    n_bits: int
        Number of bits for the quantization.
    qmethod: RangeEstimatorBase
        The quantizer to use, e.g. SymmetricUniformQuantizer
    init:RangeEstimatorBase
        Range estimator, e.g. CurrentMinmax
    per_channel: bool
        If true, will use a separate quantization grid for each kernel/channel.
    x_min: float or PyTorch Tensor
        The minimum value which needs to be represented.
    x_max: float or PyTorch Tensor
        The maximum value which needs to be represented.
    qprams: None or dict
        dictionary of parameters to instantiate the quantizer
    init_params: None or dict
         dictionary of parameters to instantiate the range estimator
    block_size:  None or int
        block size for block-wise quantization
    block_axis int
        reshaping axis for block-wise quantization
    """

    def __init__(
        self,
        qmethod: QuantizerBase = QMethods.symmetric_uniform.cls,
        init: RangeEstimatorBase = RangeEstimators.current_minmax.cls,
        per_channel=False,
        x_min=None,
        x_max=None,
        qparams=None,
        init_params=None,
        block_size=None,
    ):
        super().__init__()
        self.block_size = block_size
        self.state = Qstates.estimate_ranges
        self.qmethod = qmethod
        self.init = init
        self.per_channel = per_channel or block_size is not None
        self.qparams = qparams if qparams else {}
        self.init_params = init_params if init_params else {}
        self.range_estimator: Optional[RangeEstimatorBase] = None
        self._block_wise_quant = block_size is not None

        # define quantizer
        self.quantizer = self.qmethod(per_channel=self.per_channel, **qparams)
        self.quantizer.state = self.state

        # define range estimation method for quantizer initialisation
        if x_min is not None and x_max is not None:
            self.set_quant_range(x_min, x_max)
            self.fix_ranges()
        else:
            # set up the collector function to set the ranges
            self.range_estimator = self.init(
                per_channel=self.per_channel, quantizer=self.quantizer, **self.init_params
            )

    @property
    def n_bits(self):
        return self.quantizer.n_bits

    @property
    def block_wise_quant(self):
        return self._block_wise_quant

    @block_wise_quant.setter
    def block_wise_quant(self, flag: bool):
        self._block_wise_quant = flag

    def estimate_ranges(self):
        self.state = Qstates.estimate_ranges
        self.quantizer.state = self.state

    def fix_ranges(self):
        if self.quantizer.is_initialized:
            self.state = Qstates.fix_ranges
            self.quantizer.state = self.state
            self.quantizer.fix_ranges()
        else:
            raise QuantizerNotInitializedError()

    def learn_ranges(self):
        self.quantizer.make_range_trainable()
        self.state = Qstates.learn_ranges
        self.quantizer.state = self.state

    def estimate_ranges_train(self):
        self.state = Qstates.estimate_ranges_train
        self.quantizer.state = self.state

    def reset_ranges(self):
        if self.range_estimator is not None:
            self.range_estimator.reset()
        self.quantizer.reset()
        self.estimate_ranges()

    def reshape_tensor(self, x: torch.Tensor):
        # This function reshapes the tensor in block and only works for 2D tensors. It will raise
        # an error for 4D tensors, as it's not been tested appropriately.
        # If the tensor is 2D (C_out, C_in) it will be reshaped into (C_out * C_in // block_size,
        # block_size)

        if x.dim() == 1:
            warnings.warn(
                f"The tensor is 1-dimensional so we resort to per-channel quantization. "
                f"Consider keeping this layer in FP32"
            )
            self.block_wise_quant = False
            return x

        if x.dim() > 2:
            raise ValueError("Block-wise quantization only supported for 2D tensors for now")

        if x.shape[1] <= self.block_size:
            warnings.warn(
                f"The tensor has fewer input channels than the specified block-size. "
                f"Resort to per-channel quantization {x.shape[1]}<={self.block_size}"
            )
            self.block_wise_quant = False
            return x

        return x.view(-1, self.block_size)

    def forward(self, x):
        if self.block_wise_quant:
            # reshape tensor for block-wise quantization
            x_org_shape = x.shape
            x = self.reshape_tensor(x)

        if self.state == Qstates.estimate_ranges or (
            self.state == Qstates.estimate_ranges_train and self.training
        ):
            # Note this can be per tensor or per channel
            if self.range_estimator is None:
                raise ValueError(
                    "QuantizationManager.state indicates range setting, "
                    "but QuantizationManager.range_estimator is None"
                )
            cur_xmin, cur_xmax = self.range_estimator(x)
            self.set_quant_range(cur_xmin, cur_xmax)

        x_quant = self.quantizer(x)

        if self.block_wise_quant:
            # reshape tensor back to its original size
            x_quant = x_quant.view(x_org_shape)

        return x_quant

    def set_quant_range(self, x_min, x_max):
        self.quantizer.set_quant_range(x_min, x_max)

    def extra_repr(self):
        return "state={}".format(self.state.name)
