# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
from torch.autograd import Function


class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


class ScaleGradient(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad * ctx.scale, None


round_ste_func = RoundStraightThrough.apply
scale_grad_func = ScaleGradient.apply
