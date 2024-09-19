"""
Geometric transforms
"""
from collections.abc import Sequence
from functools import lru_cache

import numpy
from numpy.typing import NDArray, ArrayLike
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

    Args:
        offsets: Nx2 array, rows are (x, y) offsets
        pivot: (x, y) location to rotate around
        angle: rotation angle in radians

    Returns:
        Nx2 ndarray of (x, y) position after the rotation is applied.
    """
    offsets -= pivot
    offsets[:] = (rotation_matrix_2d(angle) @ offsets.T).T
    offsets += pivot
    return offsets


def apply_transforms(
        outer: ArrayLike,
        inner: ArrayLike,
        tensor: bool = False,
        ) -> NDArray[numpy.float64]:
    """
    Apply a set of transforms (`outer`) to a second set (`inner`).
    This is used to find the "absolute" transform for nested `Ref`s.

    The two transforms should be of shape Ox4 and Ix4.
    Rows should be of the form `(x_offset, y_offset, rotation_ccw_rad, mirror_across_x)`.
    The output will be of the form (O*I)x4 (if `tensor=False`) or OxIx4 (`tensor=True`).

    Args:
        outer: Transforms for the container refs. Shape Ox4.
        inner: Transforms for the contained refs. Shape Ix4.
        tensor: If `True`, an OxIx4 array is returned, with `result[oo, ii, :]` corresponding
            to the `oo`th `outer` transform applied to the `ii`th inner transform.
            If `False` (default), this is concatenated into `(O*I)x4` to allow simple
            chaining into additional `apply_transforms()` calls.

    Returns:
        OxIx4 or (O*I)x4 array. Final dimension is
            `(total_x, total_y, total_rotation_ccw_rad, net_mirrored_x)`.
    """
    outer = numpy.atleast_2d(outer).astype(float, copy=False)
    inner = numpy.atleast_2d(inner).astype(float, copy=False)

    # If mirrored, flip y's
    xy_mir = numpy.tile(inner[:, :2], (outer.shape[0], 1, 1))   # dims are outer, inner, xyrm
    xy_mir[outer[:, 3].astype(bool), :, 1] *= -1

    rot_mats = [rotation_matrix_2d(angle) for angle in outer[:, 2]]
    xy = numpy.einsum('ort,oit->oir', rot_mats, xy_mir)

    tot = numpy.empty((outer.shape[0], inner.shape[0], 4))
    tot[:, :, :2] = outer[:, None, :2] + xy
    tot[:, :, 2:] = outer[:, None, 2:] + inner[None, :, 2:]     # sum rotations and mirrored
    tot[:, :, 2] %= 2 * pi        # clamp rot
    tot[:, :, 3] %= 2             # clamp mirrored

    if tensor:
        return tot
    return numpy.concatenate(tot)
