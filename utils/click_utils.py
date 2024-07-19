# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
import click

from utils import DotDict
from utils.enums import BaseEnumOptions


class ClickEnumOption(click.Choice):
    """
    Adjusted click.Choice type for BaseOption which is based on Enum.
    """

    def __init__(self, enum_options, case_sensitive=True):
        assert issubclass(enum_options, BaseEnumOptions)
        self.base_option = enum_options
        super().__init__(self.base_option.list_names(), case_sensitive)

    def convert(self, value, param, ctx):
        # Exact match
        if value in self.choices:
            return self.base_option[value]

        # Match through normalization and case sensitivity
        # first do token_normalize_func, then lowercase
        # preserve original `value` to produce an accurate message in
        # `self.fail`
        normed_value = value
        normed_choices = self.choices

        if ctx is not None and ctx.token_normalize_func is not None:
            normed_value = ctx.token_normalize_func(value)
            normed_choices = [ctx.token_normalize_func(choice) for choice in self.choices]

        if not self.case_sensitive:
            normed_value = normed_value.lower()
            normed_choices = [choice.lower() for choice in normed_choices]

        if normed_value in normed_choices:
            return self.base_option[normed_value]

        self.fail(
            "invalid choice: %s. (choose from %s)" % (value, ", ".join(self.choices)), param, ctx
        )


def split_dict(src: dict, include=()) -> dict:
    """
    Splits dictionary into a DotDict and a remainder.
    The arguments to be placed in the first DotDict are those listed in `include`.

    Parameters
    ----------
    src: dict
        The source dictionary.
    include:
        List of keys to be returned in the first DotDict.
    """
    result = DotDict()

    for arg in include:
        result[arg] = src[arg]
    remainder = {key: val for key, val in src.items() if key not in include}
    return result, remainder
