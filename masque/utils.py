"""
Various helper functions
"""

from typing import Any, Union, Tuple

import numpy

# Type definitions
vector2 = Union[numpy.ndarray, Tuple[float, float]]


def is_scalar(var: Any) -> bool:
    """
    Alias for 'not hasattr(var, "__len__")'

    :param var: Checks if var has a length.
    """
    return not hasattr(var, "__len__")


def get_bit(bit_string: Any, bit_id: int) -> bool:
    """
    Returns true iff bit number 'bit_id' from the right of 'bitstring' is 1

    :param bit_string: st
    :param bit_id:
    :return: value of the requested bit (bool)
    """
    return bit_string & (1 << bit_id) != 0


def rotation_matrix_2d(theta: float) -> numpy.ndarray:
    """
    2D rotation matrix for rotating counterclockwise around the origin.

    :param theta: Angle to rotate, in radians
    :return: rotation matrix
    """
    return numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                        [numpy.sin(theta), +numpy.cos(theta)]])
