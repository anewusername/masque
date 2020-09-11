"""
Various helper functions
"""
from typing import Any, Union, Tuple, Sequence, Dict, List
from abc import ABCMeta

import numpy        # type: ignore


# Type definitions
vector2 = Union[numpy.ndarray, Tuple[float, float], Sequence[float]]
layer_t = Union[int, Tuple[int, int], str]
annotations_t = Dict[str, List[Union[int, float, str]]]


def is_scalar(var: Any) -> bool:
    """
    Alias for 'not hasattr(var, "__len__")'

    Args:
        var: Checks if `var` has a length.
    """
    return not hasattr(var, "__len__")


def get_bit(bit_string: Any, bit_id: int) -> bool:
    """
    Interprets bit number `bit_id` from the right (lsb) of `bit_string` as a boolean

    Args:
        bit_string: Bit string to test
        bit_id: Bit number, 0-indexed from the right (lsb)

    Returns:
        Boolean value of the requested bit
    """
    return bit_string & (1 << bit_id) != 0


def set_bit(bit_string: Any, bit_id: int, value: bool) -> Any:
    """
    Returns `bit_string`, with bit number `bit_id` set to boolean `value`.

    Args:
        bit_string: Bit string to alter
        bit_id: Bit number, 0-indexed from right (lsb)
        value: Boolean value to set bit to

    Returns:
        Altered `bit_string`
    """
    mask = (1 << bit_id)
    bit_string &= ~mask
    if value:
        bit_string |= mask
    return bit_string


def rotation_matrix_2d(theta: float) -> numpy.ndarray:
    """
    2D rotation matrix for rotating counterclockwise around the origin.

    Args:
        theta: Angle to rotate, in radians

    Returns:
        rotation matrix
    """
    return numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                        [numpy.sin(theta), +numpy.cos(theta)]])


def normalize_mirror(mirrored: Sequence[bool]) -> Tuple[bool, float]:
    """
    Converts 0-2 mirror operations `(mirror_across_x_axis, mirror_across_y_axis)`
    into 0-1 mirror operations and a rotation

    Args:
        mirrored: `(mirror_across_x_axis, mirror_across_y_axis)`

    Returns:
        `mirror_across_x_axis` (bool) and
        `angle_to_rotate` in radians
    """

    mirrored_x, mirrored_y = mirrored
    mirror_x = (mirrored_x != mirrored_y) #XOR
    angle = numpy.pi if mirrored_y else 0
    return mirror_x, angle


def remove_duplicate_vertices(vertices: numpy.ndarray, closed_path: bool = True) -> numpy.ndarray:
    """
    Given a list of vertices, remove any consecutive duplicates.

    Args:
        vertices: `[[x0, y0], [x1, y1], ...]`
        closed_path: If True, `vertices` is interpreted as an implicity-closed path
            (i.e. the last vertex will be removed if it is the same as the first)

    Returns:
        `vertices` with no consecutive duplicates.
    """
    duplicates = (vertices == numpy.roll(vertices, 1, axis=0)).all(axis=1)
    if not closed_path:
        duplicates[0] = False
    return vertices[~duplicates]


def remove_colinear_vertices(vertices: numpy.ndarray, closed_path: bool = True) -> numpy.ndarray:
    """
    Given a list of vertices, remove any superflous vertices (i.e.
        those which lie along the line formed by their neighbors)

    Args:
        vertices: Nx2 ndarray of vertices
        closed_path: If `True`, the vertices are assumed to represent an implicitly
           closed path. If `False`, the path is assumed to be open. Default `True`.

    Returns:
        `vertices` with colinear (superflous) vertices removed.
    """
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


class AutoSlots(ABCMeta):
    """
    Metaclass for automatically generating __slots__ based on superclass type annotations.

    Superclasses must set `__slots__ = ()` to make this work properly.

    This is a workaround for the fact that non-empty `__slots__` can't be used
    with multiple inheritance. Since we only use multiple inheritance with abstract
    classes, they can have empty `__slots__` and their attribute type annotations
    can be used to generate a full `__slots__` for the concrete class.
    """
    def __new__(cls, name, bases, dctn):
        parents = set()
        for base in bases:
            parents |= set(base.mro())

        slots = tuple(dctn.get('__slots__', tuple()))
        for parent in parents:
            if not hasattr(parent, '__annotations__'):
                continue
            slots += tuple(getattr(parent, '__annotations__').keys())

        dctn['__slots__'] = slots
        return super().__new__(cls, name, bases, dctn)

