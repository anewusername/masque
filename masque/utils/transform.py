"""
Geometric transforms
"""
from typing import Sequence
from functools import lru_cache

import numpy
from numpy.typing import NDArray
from numpy import pi


@lru_cache
def rotation_matrix_2d(theta: float) -> NDArray[numpy.float64]:
    """
    2D rotation matrix for rotating counterclockwise around the origin.

    Args:
        theta: Angle to rotate, in radians

    Returns:
        rotation matrix
    """
    arr = numpy.array([[numpy.cos(theta), -numpy.sin(theta)],
                       [numpy.sin(theta), +numpy.cos(theta)]])

    # If this was a manhattan rotation, round to remove some inacuraccies in sin & cos
    if numpy.isclose(theta % (pi / 2), 0):
        arr = numpy.round(arr)

    arr.flags.writeable = False
    return arr


def normalize_mirror(mirrored: Sequence[bool]) -> tuple[bool, float]:
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
    mirror_x = (mirrored_x != mirrored_y)  # XOR
    angle = numpy.pi if mirrored_y else 0
    return mirror_x, angle


def rotate_offsets_around(
        offsets: NDArray[numpy.float64],
        pivot: NDArray[numpy.float64],
        angle: float,
        ) -> NDArray[numpy.float64]:
    """
    Rotates offsets around a pivot point.
    """
    offsets -= pivot
    offsets[:] = (rotation_matrix_2d(angle) @ offsets.T).T
    offsets += pivot
    return offsets
