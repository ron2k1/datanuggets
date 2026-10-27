"""Warnings and errors raised by datanuggets."""

from __future__ import annotations

from collections.abc import Callable


class DataNuggetWarning(UserWarning):
    """A condition R reports with warning(), e.g. asking for as many nuggets as rows."""


class FlowError(ValueError):
    """A bad setting or input in the flow pipeline. A message may name its setting once as
    {param}; str() gives the setting's name and cli() the CLI flag. Messages that hold user text
    put it in before raising, and only {param} is replaced, so braces the user typed stay."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.message, self.param = message, param

    def __str__(self) -> str:
        return self.cli(lambda name: name)

    def cli(self, flag: Callable[[str], str]) -> str:
        """The message with its setting written as flag(param), e.g. by cli_flag."""
        return self.message.replace("{param}", flag(self.param)) if self.param else self.message
