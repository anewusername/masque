"""
Geometric transforms
"""
from typing import Sequence, Tuple

import numpy
from numpy.typing import NDArray


def rotation_matrix_2d(theta: float) -> NDArray[numpy.float64]:
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
    mirror_x = (mirrored_x != mirrored_y)  # XOR
    angle = numpy.pi if mirrored_y else 0
    return mirror_x, angle
