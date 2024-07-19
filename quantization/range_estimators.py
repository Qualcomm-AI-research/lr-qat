# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import copy
from abc import ABCMeta, abstractmethod
from enum import auto
from functools import partial

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import nn

from utils import to_numpy
from utils.enums import BaseEnumOptions, ClassEnumOptions, MethodMap


class RangeEstimatorBase(nn.modules.lazy.LazyModuleMixin, torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, per_channel=False, quantizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("current_xmin", nn.parameter.UninitializedBuffer())
        self.register_buffer("current_xmax", nn.parameter.UninitializedBuffer())
        self.per_channel = per_channel
        self.quantizer = quantizer

    def initialize_parameters(self, data, *args, **kwargs):  # pylint:
        # disable=arguments-differ
        num_channels = data.shape[0] if self.per_channel else 1
        with torch.no_grad():
            self.current_xmin = torch.tensor([torch.inf] * num_channels, device=data.device)
            self.current_xmax = torch.tensor([-torch.inf] * num_channels, device=data.device)

    @abstractmethod
    def forward(self, x):
        """
        Accepts an input tensor, updates the current estimates of x_min and x_max
        and retruns them.
        Parameters
        ----------
        x:  Input tensor

        Returns
        -------
        self.current_xmin: tensor

        self.current_xmax: tensor

        """
        pass

    def reset(self):
        """
        Reset the range estimator.
        """
        self.register_buffer("current_xmin", nn.parameter.UninitializedBuffer())
        self.register_buffer("current_xmax", nn.parameter.UninitializedBuffer())

        # Clear existing hooks before creating again:
        self._forward_pre_hooks.clear()
        self._load_state_dict_pre_hooks.clear()

        # Return the module in its lazy state so that thn buffer can be re-initialized
        self._initialize_hook = self.register_forward_pre_hook(self._infer_parameters)
        self._load_hook = self._register_load_state_dict_pre_hook(self._lazy_load_hook)

    def __repr__(self):
        # We overwrite this from nn.Module as we do not want to have submodules such as
        # self.quantizer in the reproduce. Otherwise it behaves as expected for an nn.Module.
        lines = self.extra_repr().split("\n")
        extra_str = lines[0] if len(lines) == 1 else "\n  " + "\n  ".join(lines) + "\n"

        return self._get_name() + "(" + extra_str + ")"


class CurrentMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, percentile=None, *args, **kwargs):
        self.percentile = percentile
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.per_channel:
            x = x.view(x.shape[0], -1)
        if self.percentile:
            axis = -1 if self.per_channel else None
            data_np = to_numpy(x)
            x_min, x_max = np.percentile(
                data_np, (self.percentile, 100 - self.percentile), axis=axis
            )
            self.current_xmin = torch.tensor(x_min).to(x.device, dtype=x.dtype)
            self.current_xmax = torch.tensor(x_max).to(x.device, dtype=x.dtype)
        else:
            self.current_xmin = x.min(-1)[0].detach() if self.per_channel else x.min().detach()
            self.current_xmax = x.max(-1)[0].detach() if self.per_channel else x.max().detach()

        return self.current_xmin, self.current_xmax


class RunningMinMaxEstimator(RangeEstimatorBase):
    def __init__(self, momentum=0.9, *args, **kwargs):
        self.momentum = momentum
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.per_channel:
            # Along 1st dim
            x_flattened = x.view(x.shape[0], -1)
            x_min = x_flattened.min(-1)[0].detach()
            x_max = x_flattened.max(-1)[0].detach()
        else:
            x_min = torch.min(x).detach()
            x_max = torch.max(x).detach()

        if self.current_xmin.isinf().any():
            self.current_xmin = x_min
            self.current_xmax = x_max
        else:
            self.current_xmin = (1 - self.momentum) * x_min + self.momentum * self.current_xmin
            self.current_xmax = (1 - self.momentum) * x_max + self.momentum * self.current_xmax

        return self.current_xmin, self.current_xmax


class OptMethod(BaseEnumOptions):
    grid = auto()
    golden_section = auto()


class OptimizationEstimatorBase(RangeEstimatorBase):
    """
    A generic range estimation class that set's quantization range by minimizing the given loss function
    between quantized and un-quantized tensor. It also implements generic 1D and 2D optimization strategies.

    """

    def __init__(
        self, num_candidates=100, opt_method=OptMethod.grid, range_margin=0.5, *args, **kwargs
    ):

        super().__init__(*args, **kwargs)
        assert opt_method in OptMethod
        self.opt_method = opt_method
        self.num_candidates = num_candidates
        self.register_buffer("loss_array", nn.parameter.UninitializedBuffer())
        self.register_buffer("max_pos_thr", nn.parameter.UninitializedBuffer())
        self.register_buffer("max_neg_thr", nn.parameter.UninitializedBuffer())
        self.register_buffer("max_search_range", nn.parameter.UninitializedBuffer())
        self.one_sided_dist = None
        self.range_margin = range_margin
        if self.quantizer is None:
            raise NotImplementedError(
                "A Quantizer must be given as an argument to the MSE Range" "Estimator"
            )
        # For asymmetric quantization
        self.max_int_skew = int(
            torch.div((2**self.quantizer.n_bits), 4, rounding_mode="floor").item()
        )

    @abstractmethod
    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False, export_numpy=False):
        """
        A loss function between quantized and un-quantized `data` that is used for range estimation.

        Parameters
        ----------
        data: torch.Tensor
            Input data.
        neg_thr, pos_thr: float or torch.Tensor
            Quantization range limits.
        per_channel_loss: bool
            Whether to output a loss for each channel separately (per-channel quantization).
        export_numpy: bool
            Whether to convert the output to a numpy array.

        Returns
        -------
        loss: 0-d or 1-d numpy or torch array (depending on the arguments)

        """
        pass

    @property
    def step_size(self):
        if self.one_sided_dist is None:
            raise NoDataPassedError()

        return self.max_search_range / self.num_candidates

    @property
    def optimization_method(self):
        if self.one_sided_dist is None:
            raise NoDataPassedError()

        if self.opt_method == OptMethod.grid:
            # Grid search method
            if self.one_sided_dist or self.quantizer.symmetric:
                # 1-D grid search
                return self._perform_1D_search
            else:
                # 2-D grid_search
                return self._perform_2D_search
        elif self.opt_method == OptMethod.golden_section:
            # Golden section method
            if self.one_sided_dist or self.quantizer.symmetric:
                return self._golden_section_symmetric
            else:
                return self._golden_section_asymmetric
        else:
            raise NotImplementedError("Optimization Method not Implemented")

    def quantize(self, x_float, x_min=None, x_max=None):
        if not self.quantizer.is_initialized:
            try:
                self.quantizer.initialize_with_data(x_float)
            except AttributeError:
                pass

        temp_q = copy.deepcopy(self.quantizer)
        if x_min is not None or x_max is not None:
            temp_q.set_quant_range(x_min, x_max)
        return temp_q(x_float)

    def golden_sym_loss(self, range, data):
        """
        Loss function passed to the golden section optimizer from scipy in case of symmetric
        quantization
        """
        neg_thr = 0 if self.one_sided_dist else -range
        pos_thr = range
        return self.loss_fx(data, neg_thr, pos_thr, export_numpy=True)

    def golden_asym_shift_loss(self, shift, range, data):
        """
        Inner Loss function (shift) passed to the golden section optimizer from scipy
        in case of asymmetric quantization
        """
        pos_thr = range + shift
        neg_thr = -range + shift
        return self.loss_fx(data, neg_thr, pos_thr, export_numpy=True)

    def golden_asym_range_loss(self, range, data):
        """
        Outer Loss function (range) passed to the golden section optimizer from scipy in case of
         asymmetric quantization
        """
        temp_delta = 2 * range / (2**self.quantizer.n_bits - 1)
        max_shift = temp_delta * self.max_int_skew
        result = minimize_scalar(
            self.golden_asym_shift_loss,
            args=(range, data),
            bounds=(-max_shift, max_shift),
            method="Bounded",
        )
        return result.fun

    def _define_search_range(self, data):
        if self.per_channel:
            data = data.view(data.shape[0], -1)
        else:
            data = data.view(1, -1)
        self.channel_groups = len(data) if self.per_channel else 1
        self.current_xmax = torch.zeros(self.channel_groups, device=data.device, dtype=data.dtype)
        self.current_xmin = torch.zeros(self.channel_groups, device=data.device, dtype=data.dtype)

        if self.one_sided_dist or self.quantizer.symmetric:
            # 1D search space
            self.loss_array = torch.zeros(
                (self.channel_groups, self.num_candidates + 1), device=data.device
            )  # 1D search
            # space
            self.loss_array[:, 0] = torch.inf  # exclude interval_start=interval_finish
            # Defining the search range for clipping thresholds
            self.max_pos_thr = (
                data.view(self.channel_groups, -1).abs().max(1)[0] + self.range_margin
            )
            self.max_neg_thr = -self.max_pos_thr
            self.max_search_range = self.max_pos_thr
        else:
            # 2D search space (3rd and 4th index correspond to asymmetry where fourth
            # index represents whether the skew is positive (0) or negative (1))
            self.loss_array = torch.zeros(
                (int(self.channel_groups), int(self.num_candidates) + 1, int(self.max_int_skew), 2),
                device=data.device,
            )  # 2D search
            # space
            self.loss_array[:, 0, :, :] = torch.inf  # exclude interval_start=interval_finish
            # Define the search range for clipping thresholds in asymmetric case
            self.max_pos_thr = data.view(self.channel_groups, -1).max(1)[0] + self.range_margin
            self.max_neg_thr = data.view(self.channel_groups, -1).min(1)[0] - self.range_margin
            self.max_search_range = torch.max(self.max_pos_thr.abs(), self.max_neg_thr.abs())

    def _perform_1D_search(self, data):
        """
        Grid search through all candidate quantizers in 1D to find the best
        The loss is accumulated over all batches without any momentum
        :param data: input tensor
        """
        for cand_index in range(1, self.num_candidates + 1):
            neg_thr = (
                torch.zeros_like(self.step_size)
                if self.one_sided_dist
                else -self.step_size * cand_index
            )
            pos_thr = self.step_size * cand_index
            self.loss_array[:, cand_index] += self.loss_fx(
                data, neg_thr, pos_thr, per_channel_loss=self.per_channel
            )

        _, min_cand = self.loss_array.min(dim=1)
        self.current_xmin = (
            torch.zeros_like(self.step_size) if self.one_sided_dist else -self.step_size * min_cand
        )
        self.current_xmax = self.step_size * min_cand

    def _perform_2D_search(self, data):
        """
        Grid search through all candidate quantizers in 1D to find the best
        The loss is accumulated over all batches without any momentum
        Parameters
        ----------
        data:   PyTorch Tensor
        Returns
        -------

        """
        for cand_index in range(1, self.num_candidates + 1):
            # defining the symmetric quantization range
            temp_start = -self.step_size * cand_index
            temp_finish = self.step_size * cand_index
            temp_delta = (temp_finish - temp_start) / (2**self.quantizer.n_bits - 1)
            for shift in range(self.max_int_skew):
                for reverse in range(2):
                    # introducing asymmetry in the quantization range
                    skew = ((-1) ** reverse) * shift * temp_delta
                    neg_thr = max(temp_start + skew, self.max_neg_thr)
                    pos_thr = min(temp_finish + skew, self.max_pos_thr)

                    self.loss_array[:, cand_index, shift, reverse] += self.loss_fx(
                        data, neg_thr, pos_thr, per_channel_loss=self.per_channel
                    )

        for channel_index in range(self.channel_groups):
            min_cand, min_shift, min_reverse = np.unravel_index(
                np.argmin(self.loss_array[channel_index], axis=None),
                self.loss_array[channel_index].shape,
            )
            min_interval_start = -self.step_size * min_cand
            min_interval_finish = self.step_size * min_cand
            min_delta = float(min_interval_finish - min_interval_start) / (
                2**self.quantizer.n_bits - 1
            )
            min_skew = ((-1) ** min_reverse) * min_shift * min_delta
            xmin = max(min_interval_start + min_skew, self.max_neg_thr)
            xmax = min(min_interval_finish + min_skew, self.max_pos_thr)

            self.current_xmin[channel_index] = torch.tensor(xmin).to(
                device=data.device, dtype=data.dtype
            )
            self.current_xmax[channel_index] = torch.tensor(xmax).to(
                device=data.device, dtype=data.dtype
            )

    def _golden_section_symmetric(self, data):
        for channel_index in range(self.channel_groups):
            if channel_index == 0 and not self.per_channel:
                data_segment = data
                max_search_range = self.max_search_range.item()
            else:
                data_segment = data[channel_index]
                max_search_range = float(self.max_search_range[channel_index])

            self.result = minimize_scalar(
                self.golden_sym_loss,
                args=data_segment,
                bounds=(0.01 * max_search_range, max_search_range),
                method="Bounded",
            )
            self.current_xmax[channel_index] = torch.tensor(self.result.x).to(
                device=data.device, dtype=data.dtype
            )
            self.current_xmin[channel_index] = (
                torch.tensor(0.0).to(device=data.device, dtype=data.dtype)
                if self.one_sided_dist
                else -self.current_xmax[channel_index]
            )

    def _golden_section_asymmetric(self, data):
        for channel_index in range(self.channel_groups):
            if channel_index == 0 and not self.per_channel:
                data_segment = data
                max_search_range = self.max_search_range.item()
            else:
                data_segment = data[channel_index]
                max_search_range = float(self.max_search_range[channel_index])

            self.result = minimize_scalar(
                self.golden_asym_range_loss,
                args=data_segment,
                bounds=(0.01 * max_search_range, max_search_range),
                method="Bounded",
            )
            self.final_range = self.result.x
            temp_delta = 2 * self.final_range / (2**self.quantizer.n_bits - 1)
            max_shift = temp_delta * self.max_int_skew
            self.subresult = minimize_scalar(
                self.golden_asym_shift_loss,
                args=(self.final_range, data_segment),
                bounds=(-max_shift, max_shift),
                method="Bounded",
            )
            self.final_shift = self.subresult.x
            self.current_xmax[channel_index] = torch.tensor(self.final_range + self.final_shift).to(
                device=data.device, dtype=data.dtype
            )
            self.current_xmin[channel_index] = torch.tensor(
                -self.final_range + self.final_shift
            ).to(device=data.device, dtype=data.dtype)

    def initialize_parameters(self, data, *args, **kwargs):  # pylint: disable=arguments-differ
        self.one_sided_dist = bool((data.min() >= 0).item())
        self._define_search_range(data)

    def forward(self, data):
        org_dtype = data.dtype
        data = data.to(torch.float32)
        self.optimization_method(data)
        if self.opt_method == OptMethod.grid and self.per_channel:
            # delete the loss array can get very big for per-channel/block
            # (note: per-channel/block is only used for weights thus keeping it over multiple batches is not needed)
            print("Reset loss array")
            self.register_buffer("loss_array", nn.parameter.UninitializedBuffer())

        return self.current_xmin.to(org_dtype), self.current_xmax.to(org_dtype)

    def reset(self):
        super().reset()
        self.one_sided_dist = None
        self.register_buffer("loss_array", nn.parameter.UninitializedBuffer())
        self.register_buffer("max_pos_thr", nn.parameter.UninitializedBuffer())
        self.register_buffer("max_neg_thr", nn.parameter.UninitializedBuffer())
        self.register_buffer("max_search_range", nn.parameter.UninitializedBuffer())

    def extra_repr(self):
        repr = "opt_method={}".format(self.opt_method.name)
        if self.opt_method == OptMethod.grid:
            repr += " ,num_candidates={}".format(self.num_candidates)
        return repr


class LPnorm_Estimator(OptimizationEstimatorBase):
    def __init__(self, norm=2.0, *args, **kwargs):
        """Range estimator that optimizes any LP-norm on the given data."""
        super().__init__(*args, **kwargs)
        self.norm = norm

    def loss_fx(self, data, neg_thr, pos_thr, per_channel_loss=False, export_numpy=False):
        y = self.quantize(data, x_min=neg_thr, x_max=pos_thr)
        temp_sum = torch.sum(((data - y).abs() ** self.norm).view(len(data), -1), dim=1)
        # if we want to return the MSE loss of each channel separately, speeds up the per-channel
        # grid search
        if not per_channel_loss:
            temp_sum = temp_sum.sum()

        if export_numpy:
            return to_numpy(temp_sum)

        return temp_sum


class MSE_Estimator(LPnorm_Estimator):
    """This one is not really needed, but rather for backwards compatability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, norm=2.0, **kwargs)


class NoDataPassedError(Exception):
    """Raised data has been passed inot the Range Estimator"""

    def __init__(self):
        super().__init__("Data must be pass through the range estimator to be initialized")


class RangeEstimators(ClassEnumOptions):
    current_minmax = MethodMap(CurrentMinMaxEstimator)
    running_minmax = MethodMap(RunningMinMaxEstimator)
    MSE = MethodMap(MSE_Estimator)
    LP_p24 = MethodMap(partial(LPnorm_Estimator, norm=2.4, range_margin=0))
    LP_p3 = MethodMap(partial(LPnorm_Estimator, norm=3.0, range_margin=0))
    LP_p35 = MethodMap(partial(LPnorm_Estimator, norm=3.5, range_margin=0))
    LP_p4 = MethodMap(partial(LPnorm_Estimator, norm=4.0, range_margin=0))
    LP_p5 = MethodMap(partial(LPnorm_Estimator, norm=5.0, range_margin=0))
