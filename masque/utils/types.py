"""
Type definitions
"""
from typing import Union, Tuple, Dict, List, Protocol


layer_t = Union[int, Tuple[int, int], str]
annotations_t = Dict[str, List[Union[int, float, str]]]


class SupportsBool(Protocol):
    def __bool__(self) -> bool:
        ...
