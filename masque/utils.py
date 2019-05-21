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


def normalize_mirror(mirrored: Tuple[bool, bool]) -> Tuple[bool, float]:
    mirrored_x, mirrored_y = mirrored
    mirror_x = (mirrored_x != mirrored_y) #XOR
    angle = numpy.pi if mirrored_y else 0
    return mirror_x, angle


def remove_duplicate_vertices(vertices: numpy.ndarray, closed_path: bool = True) -> numpy.ndarray:
        duplicates = (vertices == numpy.roll(vertices, 1, axis=0)).all(axis=1)
        if not closed_path:
            duplicates[0] = False
        return vertices[~duplicates]


def remove_colinear_vertices(vertices: numpy.ndarray, closed_path: bool = True) -> numpy.ndarray:
        '''
        Given a list of vertices, remove any superflous vertices (i.e.
            those which lie along the line formed by their neighbors)

        :param vertices: Nx2 ndarray of vertices
        :param closed_path: If True, the vertices are assumed to represent an implicitly
            closed path. If False, the path is assumed to be open. Default True.
        :return:
        '''
        vertices = numpy.array(vertices)

        # Check for dx0/dy0 == dx1/dy1

        dv = numpy.roll(vertices, -1, axis=0) - vertices #       [y1-y0, y2-y1, ...]
        dxdy = dv * numpy.roll(dv, 1, axis=0)[:, ::-1]   #[[dx0*(dy_-1), (dx_-1)*dy0], dx1*dy0, dy1*dy0]]

        dxdy_diff = numpy.abs(numpy.diff(dxdy, axis=1))[:, 0]
        err_mult = 2 * numpy.abs(dxdy).sum(axis=1) + 1e-40

        slopes_equal = (dxdy_diff / err_mult) < 1e-15
        if not closed_path:
            slopes_equal[[0, -1]] = False

        return vertices[~slopes_equal]
