# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
from torch import nn


class QuantizerBase(nn.Module):
    """
    Arguments:
        n_bits: number of bits for quantizer
        per_channel: use per-channel quantization if true, otherwise per-tensro
        act_quant: flag to indicate whether current quantizer is used to quantize activations.
        *_ and **__: used to absorb arguments not used by subclass implementations (e.g. scale_domain,
        discretizer, etc.)
    """

    def __init__(self, n_bits, per_channel=False, act_quant=False, *_, **__):

        super().__init__()
        self.n_bits = n_bits
        self.act_quant = act_quant
        self.per_channel = per_channel
        self.state = None
        self.x_min_fp32 = self.x_max_fp32 = None

    @property
    def is_initialized(self):
        raise NotImplementedError()

    @property
    def x_max(self):
        raise NotImplementedError()

    @property
    def symmetric(self):
        raise NotImplementedError()

    @property
    def x_min(self):
        raise NotImplementedError()

    @property
    def effective_bit_width(self):
        return self.n_bits

    def forward(self, x_float):
        raise NotImplementedError()

    def _adjust_params_per_channel(self, x):
        raise NotImplementedError()

    def set_quant_range(self, x_min, x_max):
        raise NotImplementedError()

    def reset_nbits(self, n_bits):
        assert self.x_max_fp32 is not None
        self.n_bits = n_bits
        self.set_quant_range(self.x_min_fp32, self.x_max_fp32)

    def extra_repr(self):
        return "n_bits={}, per_channel={}, is_initalized={}".format(
            self.n_bits, self.per_channel, self.is_initialized
        )

    def reset(self):
        self._delta = None

    def fix_ranges(self):
        raise NotImplementedError()

    def make_range_trainable(self):
        raise NotImplementedError()
