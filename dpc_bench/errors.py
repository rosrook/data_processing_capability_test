"""Shared errors."""

from __future__ import annotations


class LLMError(RuntimeError):
    pass


class LLMConfigError(LLMError):
    pass


class TaskError(RuntimeError):
    pass
