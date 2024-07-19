# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
from quantization.quantizers.uniform_quantizers import (
    AsymmetricUniformQuantizer,
    DynamicAsymmetricUniformQuantizer,
    SymmetricUniformQuantizer,
)
from utils.enums import ClassEnumOptions, MethodMap


class QMethods(ClassEnumOptions):
    symmetric_uniform = MethodMap(SymmetricUniformQuantizer)
    asymmetric_uniform = MethodMap(AsymmetricUniformQuantizer)
    dynamic_asymmetric_uniform = MethodMap(DynamicAsymmetricUniformQuantizer)
