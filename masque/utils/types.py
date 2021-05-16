"""
Type definitions
"""
from typing import Union, Tuple, Sequence, Dict, List


layer_t = Union[int, Tuple[int, int], str]
annotations_t = Dict[str, List[Union[int, float, str]]]
