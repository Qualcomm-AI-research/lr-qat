# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
from collections import namedtuple
from enum import Flag, auto
from functools import partial


class BaseEnumOptions(Flag):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    @property
    def cls(self):
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)


MethodMap = partial(namedtuple("MethodMap", ["value", "cls"]), auto())


class QuantSetup(BaseEnumOptions):
    all = auto()  # quantize all activations (output quantization)
    LSQ_output = auto()  # first and last layer kept at 8-bits with output quantization
    FP_logits = auto()  # logits in FP32 (output quantization)
    fc4 = auto()
    fc4_dw8 = auto()
    LSQ_input = auto()

    # LLM options
    FP_head = auto()
    FP_head_embd = auto()
    FP_head_embd_norm = auto()


class DatasetSetup(BaseEnumOptions):
    bookcorpus_and_wiki = auto()
    ptb = auto()
    wikitext_2 = auto()
    wikitext_103 = auto()
    slimpajama_wiki = auto()  # Slimpajama train, Wikipedia val.
    slimpajama_wikitext_2 = auto()  # Slimpajama train, Wikitext-2 val.
    SLIMPAJAMA = slimpajama_wiki | slimpajama_wikitext_2


class PreprocessingType(BaseEnumOptions):
    join = auto()
    join_nn = auto()
    line_by_line = auto()


class TrainableParamGroup(BaseEnumOptions):
    all = auto()
    freeze_embd = auto()  # all model params, except token embeddings
    norm_only = auto()
    none = auto()  # freeze all model params (quant params can still be trained)

    # train LoRA params
    lora = auto()
    lora_head = auto()
    lora_head_norm = auto()
    lora_head_norm_embd = auto()

    LORA = lora | lora_head | lora_head_norm | lora_head_norm_embd


class LR_QAT_Method(BaseEnumOptions):
    naive = auto()  # keep A, B unquantized (~QLoRA)

    # Methods
    lr_qat_int = auto()

    lr_qat_fp32 = auto()
    lr_qat_bf16 = auto()
    lr_qat_fp16 = auto()
    lr_qat_fixed_point8 = auto()


LR_QAT_PHI = (
    LR_QAT_Method.lr_qat_fp32,
    LR_QAT_Method.lr_qat_fp16,
    LR_QAT_Method.lr_qat_bf16,
    LR_QAT_Method.lr_qat_fixed_point8,
)


class ValidationQuantMethod(BaseEnumOptions):
    no_quant = auto()  # no quant. applied
    ptq = auto()  # vanilla PTQ method (homogeneous bitwidth)


class ActQuantBits(BaseEnumOptions):
    a8 = auto()  # all activations in 8-bit
    a8kv4 = auto()  # all activations in 8-bit, except KV cache in 4-bits
    a4 = auto()  # all activations in 4-bit
