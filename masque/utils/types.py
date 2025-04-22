"""
Type definitions
"""
from typing import Protocol


layer_t = int | tuple[int, int] | str
annotations_t = dict[str, list[int | float | str]] | None


class SupportsBool(Protocol):
    def __bool__(self) -> bool:
        ...
