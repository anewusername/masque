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
    Returns true iff bit number 'bit_id' from the right of 'bit_string' is 1

    :param bit_string: Bit string to test
    :param bit_id: Bit number, 0-indexed from the right (lsb)
    :return: value of the requested bit (bool)
    """
    return bit_string & (1 << bit_id) != 0


def set_bit(bit_string: Any, bit_id: int, value: bool) -> Any:
    """
    Returns 'bit_string' with bit number 'bit_id' set to 'value'.

    :param bit_string: Bit string to alter
    :param bit_id: Bit number, 0-indexed from right (lsb)
    :param value: Boolean value to set bit to
    :return: Altered 'bit_string'
    """
    mask = (1 << bit_id)
    bit_string &= ~mask
    if value:
        bit_string |= mask
    return bit_string


def rotation_matrix_2d(theta: float) -> numpy.ndarray:
    """
    2D rotation matrix for rotating counterclockwise around the origin.

    :param theta: Angle to rotate, in radians
    :return: rotation matrix
    """
    return numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                        [numpy.sin(theta), +numpy.cos(theta)]])
