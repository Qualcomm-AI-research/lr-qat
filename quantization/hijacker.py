# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import copy
import logging
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint
from transformers.activations import NewGELUActivation

from quantization.base_quantized_classes import QuantizedModule
from quantization.quantization_manager import QuantizationManager
from quantization.quantizers.rounding_utils import round_ste_func
from quantization.range_estimators import RangeEstimators
from utils import to_numpy
from utils.enums import LR_QAT_PHI, LR_QAT_Method

activations_set = [
    nn.ReLU,
    nn.ReLU6,
    nn.Hardtanh,
    nn.Sigmoid,
    nn.Tanh,
    nn.GELU,
    nn.PReLU,
    nn.Hardswish,
    nn.Hardsigmoid,
    nn.SiLU,
    NewGELUActivation,
    nn.SiLU,
]


class QuantizationHijacker(QuantizedModule):
    """Mixin class that 'hijacks' the forward pass in a module to perform quantization and
    dequantization on the weights and output distributions.

    Usage:
    To make a quantized nn.Linear layer:
    class HijackedLinear(QuantizationHijacker, nn.Linear):
        pass

    It is vital that QSchemeForwardHijacker is the first parent class, and that the second parent
    class derives from nn.Module, otherwise it will not be reached by a super(., .) call.

    NB: this implementation (for now) assumes that there will always be some training involved,
    e.g. to estimate the activation ranges.
    """

    def __init__(self, *args, activation: nn.Module = None, **kwargs):

        super().__init__(*args, **kwargs)
        if activation:
            assert isinstance(activation, tuple(activations_set)), str(activation)

        self.activation_function = copy.deepcopy(activation) if activation else None

        self.activation_quantizer = QuantizationManager(
            qmethod=self.act_method,
            init=self.act_range_method,
            qparams=self.act_qparams,
            init_params=self.act_range_options,
        )

        if self.weight_range_method == RangeEstimators.current_minmax:
            weight_init_params = dict(percentile=self.percentile)
        else:
            weight_init_params = self.weight_range_options

        self.weight_quantizer = QuantizationManager(
            qmethod=self.method,
            init=self.weight_range_method,
            per_channel=self.per_channel_weights,
            qparams=self.weight_qparams,
            init_params=weight_init_params,
            block_size=self.weight_block_size,
        )

        self.prune_manager = None
        if hasattr(self, "prune_method") and self.prune_method is not None:
            self.prune_manager = self.prune_method(self, **self.prune_kwargs)

        self.activation_save_target = None
        self.activation_save_name = None
        self.adaround_learned_bias_corr = None
        self.frozen_integer_weights = False
        self.scale = None

        # LoRA
        self.lora_r = 0
        self.lora_merge_weights = False
        self.lora_merged = False
        self.lora_method = None
        self.use_checkpointing = None
        self.lora_Bt: Optional[Parameter] = None
        self.lora_A: Optional[Parameter] = None
        self.applied_phi = False

    def _apply_lora_qat_w_int(self, x, bias, offsets):
        """
        Apply LoRA-QAT with `s * clip(round(W_int + AB), ...)` formulation
        """
        W_int = self.weight  # frozen W_int
        w_qm = self.weight_quantizer
        q = self.weight_quantizer.quantizer
        q_min, q_max = int(q.int_min), int(q.int_max)

        if self.lora_r > 0:
            C = (self.lora_A @ self.lora_Bt) * self.lora_scaling
            W_int_hat = W_int + round_ste_func(C)
            W_int_hat = torch.clamp(W_int_hat, q_min, q_max)
        else:
            W_int_hat = W_int

        if w_qm.block_wise_quant:
            orig_size = W_int_hat.size()
            W_hat = self.scale * w_qm.reshape_tensor(W_int_hat)
            W_hat = W_hat.view(orig_size)
        else:
            W_hat = self.scale * W_int_hat

        W_hat = W_hat.to(x.dtype)  # downcast W_hat if x has a different dtype
        res = self.run_forward(x, W_hat, bias, offsets=offsets)
        return res

    def _apply_lora_qat_w_phi(self, x, bias, offsets):
        """
        Apply LoRA-QAT with `s * clip(round(\phi(W_fp_0/s_0) + AB), ...)` formulation
        """
        Phi_0 = self.weight  # frozen \phi(W/s0)
        w_qm = self.weight_quantizer
        q = self.weight_quantizer.quantizer
        C = (self.lora_A @ self.lora_Bt) * self.lora_scaling
        s = q.scale
        q_min, q_max = int(q.int_min), int(q.int_max)

        if self.lora_method == LR_QAT_Method.lr_qat_fixed_point8:
            Phi_0 = self._fixed_point8_to_full_precision(Phi_0)  # upcast back to FP

        Z = Phi_0 + C
        W_int = torch.clamp(round_ste_func(Z), q_min, q_max)

        if w_qm.block_wise_quant:
            orig_size = W_int.size()
            W_hat = s * w_qm.reshape_tensor(W_int)
            W_hat = W_hat.view(orig_size)
        else:
            W_hat = s * W_int

        W_hat = W_hat.to(x.dtype)  # downcast W_hat if x has a different dtype
        res = self.run_forward(x, W_hat, bias, offsets=offsets)
        return res

    def _full_precision_to_fixed_point8(self, x: torch.Tensor):
        q = self.weight_quantizer.quantizer
        q_min, q_max = int(q.int_min), int(q.int_max)
        n_bits = int(q.n_bits)
        assert 2 <= n_bits <= 7
        assert q_min < 0  # for now assume symmeric signed quantization

        x = torch.clamp(x, q_min, q_max)  # e.g., [-8, 7] for n_bits == 4
        x *= 2 ** (8 - n_bits)  # [-128, 127 - 2^(8-b)]
        x = torch.round(x).to(torch.int8)  # {-128, -127, ..., 127 - 2^(8-b)} \in {-128, ..., +127}
        return x

    def _fixed_point8_to_full_precision(self, x: torch.Tensor):
        q = self.weight_quantizer.quantizer
        q_min, q_max = int(q.int_min), int(q.int_max)
        n_bits = int(q.n_bits)

        quantizer_dtype = self.weight_quantizer.quantizer.delta.dtype
        x = x.to(quantizer_dtype)
        x /= 2 ** (8 - n_bits)

        eps = 1e-8
        assert x.min() >= q_min - eps
        assert x.max() <= q_max + eps
        return x

    def forward(self, x, offsets=None):
        # Quantize input
        if self.quantize_input and self._quant_a:
            x = self.activation_quantizer(x)

        # Get quantized weight
        weight, bias = self.get_params()

        # LoRA
        if self.lora_method is not None and not self.lora_merged:
            # **
            # ** LoRA naive **
            # **
            if self.lora_method == LR_QAT_Method.naive:
                assert self.lora_r > 0

                C = (self.lora_A @ self.lora_Bt) * self.lora_scaling
                res = self.run_forward(x, weight + C, bias, offsets=offsets)

            # **
            # ** LoRA-QAT (W_int variants) **
            # **
            elif self.lora_method in (LR_QAT_Method.lr_qat_int,):
                # check
                if not self.frozen_integer_weights:
                    raise RuntimeError(
                        f'Method "{self.lora_method.name}" assumes frozen integer '
                        f"weights (--freeze-integer-weights)"
                    )

                # compute `layer(W_hat(W_int, A, B), x)`
                if self.use_checkpointing:
                    res = checkpoint(
                        self._apply_lora_qat_w_int,
                        x,
                        bias,
                        offsets,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    res = self._apply_lora_qat_w_int(x, bias, offsets)

            # **
            # ** LoRA-QAT (\phi variants) **
            # **
            elif self.lora_method in LR_QAT_PHI:
                # checks
                assert self.lora_r > 0
                if self.frozen_integer_weights:
                    raise RuntimeError(
                        f'Method "{self.lora_method.name}" assumes floating point '
                        f"weights, but --freeze-integer-weights was used"
                    )

                # store initial scale
                if not self.applied_phi:
                    w_qm = self.weight_quantizer
                    scale_0 = w_qm.quantizer.scale.detach().clone()

                    if w_qm.block_wise_quant:
                        Phi_0 = self.weight.data
                        orig_size = Phi_0.size()
                        Phi_0 = w_qm.reshape_tensor(Phi_0) / scale_0
                        Phi_0 = Phi_0.view(orig_size)
                    else:
                        Phi_0 = self.weight.data / scale_0

                    # apply down casting \phi
                    if self.lora_method == LR_QAT_Method.lr_qat_fp32:
                        pass
                    elif self.lora_method == LR_QAT_Method.lr_qat_bf16:
                        Phi_0 = Phi_0.to(torch.bfloat16)
                    elif self.lora_method == LR_QAT_Method.lr_qat_fp16:
                        Phi_0 = Phi_0.to(torch.float16)
                    elif self.lora_method == LR_QAT_Method.lr_qat_fixed_point8:
                        Phi_0 = self._full_precision_to_fixed_point8(Phi_0)
                    else:
                        raise ValueError(f"Unknown method {self.lora_method}")

                    # register the result as a new buffer
                    del self.weight
                    self.register_buffer("weight", Phi_0)

                    # mark down casting is applied
                    self.applied_phi = True

                # compute `layer(W_hat(W, A, B), x)`
                if self.use_checkpointing:
                    res = checkpoint(
                        self._apply_lora_qat_w_phi,
                        x,
                        bias,
                        offsets,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    res = self._apply_lora_qat_w_phi(x, bias, offsets)
        else:
            res = self.run_forward(x, weight, bias, offsets=offsets)

        # Apply fused activation function
        if self.activation_function is not None:
            res = self.activation_function(res)

        # Quantize output
        if not self.quantize_input and self._quant_a:
            res = self.activation_quantizer(res)
        return res

    def get_params(self):
        if not self.training and self.caching and self.cached_params:
            return self.cached_params

        weight, bias = self.get_weight_bias()

        # during training of LoRA-QAT \phi variant, skip quantization here,
        # will be done in the `forward` method
        if not self.lora_merge_weights and self.lora_method in LR_QAT_PHI:
            return weight, bias

        # merge LoRA weights, if needed
        if self.lora_merge_weights and not self.lora_merged:
            # Merge the weights and mark it
            if self.lora_r > 0:
                print("merging LoRA weights")

                q = self.weight_quantizer.quantizer
                q_min, q_max = int(q.int_min), int(q.int_max)

                # *** freeze_int_weights = True ***
                if self.frozen_integer_weights:
                    W_int = self.weight
                    C = (self.lora_A @ self.lora_Bt) * self.lora_scaling

                    # ** LoRA naive **
                    if self.lora_method == LR_QAT_Method.naive:
                        C_int = q.to_integer_forward(C).to(torch.int8)
                        merged_weight = torch.clamp(W_int + C_int, q_min, q_max)

                    # ** LoRA-QAT :: W_int + round(AB) **
                    elif self.lora_method == LR_QAT_Method.lr_qat_int:
                        C_int = torch.round(C).to(torch.int8)
                        merged_weight = torch.clamp(W_int + C_int, q_min, q_max)

                    # fuse
                    self.weight.data = merged_weight

                # *** freeze_int_weights = False ***
                else:
                    # ** LoRA-QAT (\Phi variants) **
                    if self.lora_method in LR_QAT_PHI:
                        Phi_0 = self.weight  # frozen (original) W_fp
                        C = (self.lora_A @ self.lora_Bt) * self.lora_scaling
                        s = q.scale

                        if self.lora_method == LR_QAT_Method.lr_qat_fixed_point8:
                            Phi_0 = self._fixed_point8_to_full_precision(Phi_0)  # upcast back to FP

                        Z = Phi_0 + C
                        merged_W_int = torch.round(Z).to(torch.int8)
                        merged_W_int.clamp_(q_min, q_max)

                        # register it as frozen integer weights & final scale
                        del self.weight
                        self.register_buffer("weight", merged_W_int)

                        assert self.scale is None
                        del self.scale
                        self.register_buffer("scale", s)

                        self.caching = False
                        self.frozen_integer_weights = True

                    # ** vanilla FP LoRA **
                    else:
                        self.weight.data += (self.lora_A @ self.lora_Bt) * self.lora_scaling

            # mark that LoRA weights have been merged
            self.lora_merged = True

        if self.frozen_integer_weights:
            w_qm = self.weight_quantizer
            if w_qm.block_wise_quant:
                orig_size = self.weight.size()
                weight = w_qm.reshape_tensor(self.weight) * self.scale
                weight = weight.view(orig_size)
            else:
                weight = self.weight * self.scale

            return weight, bias

        if self.prune_manager is not None:
            weight, bias = self.prune_manager(weight, bias)

        if self._quant_w:
            weight = self.quantize_weights(weight)

        # add learnable bias from AdaRound, if provided
        if self.adaround_learned_bias_corr is not None and self._quant_w:
            if bias is None:
                bias = self.adaround_learned_bias_corr
            else:
                bias = bias + self.adaround_learned_bias_corr

        if self.caching and not self.training and self.cached_params is None:
            self.cached_params = (
                torch.Tensor(to_numpy(weight)).to(weight.device).to(weight.dtype),
                (
                    torch.Tensor(to_numpy(bias)).to(bias.device).to(bias.dtype)
                    if bias is not None
                    else None
                ),
            )
        return weight, bias

    def quantize_weights(self, weights):
        return self.weight_quantizer(weights)

    def get_weight_bias(self):
        bias = None
        if hasattr(self, "bias"):
            bias = self.bias
        return self.weight, bias

    def freeze_integer_weights(self, device=None):
        # checks
        if self.frozen_integer_weights:
            return

        if not self._quant_w:
            logging.info(
                "Freeze integer weights skipped: this module did not enabled quantization of weights."
            )
            return

        # For now we save the integer weights in INT8
        w_qm = self.weight_quantizer
        q = self.weight_quantizer.quantizer

        # Use quantization manager to call potential range estimation
        wq = self.weight_quantizer(self.weight)

        if w_qm.block_wise_quant:
            orig_size = self.weight.size()
            out = w_qm.reshape_tensor(self.weight)  # reshaped
            w_int = q.to_integer_forward(out).to(torch.int8)  # still reshaped
            out = w_int * q.scale  # still reshaped
            out = out.view(orig_size)
            w_int = w_int.view(
                orig_size
            )  # need to reshape because we register this as new self.weight
            assert torch.allclose(wq, out)
        else:
            w_int = q.to_integer_forward(self.weight).to(torch.int8)
            assert torch.allclose(wq, w_int * q.scale)

        del self.weight
        self.register_buffer("weight", w_int)
        self.scale = torch.nn.Parameter(q.scale)

        if w_qm.block_wise_quant:
            orig_size = self.weight.size()
            out = w_qm.reshape_tensor(self.weight) * self.scale
            out = out.view(orig_size)
            assert torch.allclose(wq, out)
        else:
            assert torch.allclose(wq, self.weight * self.scale)

        # Some housekeeping
        self.caching = False
        self.frozen_integer_weights = True

    def run_forward(self, x, weight, bias, offsets=None):
        # Performs the actual linear operation of the layer
        raise NotImplementedError()

    def extra_repr(self):
        activation = "input" if self.quantize_input else "output"
        return f"{super().extra_repr()}-{activation}"
