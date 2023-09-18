"""
 Ref provides basic support for nesting Pattern objects within each other.
 It carries offset, rotation, mirroring, and scaling data for each individual instance.
"""
from typing import Mapping, TYPE_CHECKING, Self
import copy

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from .utils import annotations_t, rotation_matrix_2d
from .repetition import Repetition
from .traits import (
    PositionableImpl, RotatableImpl, ScalableImpl,
    Mirrorable, PivotableImpl, Copyable, RepeatableImpl, AnnotatableImpl,
    )


if TYPE_CHECKING:
    from . import Pattern


class Ref(
        PositionableImpl, RotatableImpl, ScalableImpl, Mirrorable,
        PivotableImpl, Copyable, RepeatableImpl, AnnotatableImpl,
        ):
    """
    `Ref` provides basic support for nesting Pattern objects within each other.

    It containts the transformation (mirror, rotation, scale, offset, repetition)
    and annotations for a single instantiation of a `Pattern`.

    Note that the target (i.e. which pattern a `Ref` instantiates) is not stored within the
    `Ref` itself, but is specified by the containing `Pattern`.

    Order of operations is (mirror, rotate, scale, translate, repeat).
    """
    __slots__ = (
        '_mirrored',
        # inherited
        '_offset', '_rotation', 'scale', '_repetition', '_annotations',
        )

    _mirrored: bool
    """ Whether to mirror the instance across the x axis (new_y = -old_y)ubefore rotating. """

    # Mirrored property
    @property
    def mirrored(self) -> bool:     # mypy#3004, setter should be SupportsBool
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: bool) -> None:
        self._mirrored = bool(val)

    def __init__(
            self,
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            mirrored: bool = False,
            scale: float = 1.0,
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            ) -> None:
        """
        Note: Order is (mirror, rotate, scale, translate, repeat)

        Args:
            offset: (x, y) offset applied to the referenced pattern. Not affected by rotation etc.
            rotation: Rotation (radians, counterclockwise) relative to the referenced pattern's (0, 0).
            mirrored: Whether to mirror the referenced pattern across its x axis before rotating.
            scale: Scaling factor applied to the pattern's geometry.
            repetition: `Repetition` object, default `None`
        """
        self.offset = offset
        self.rotation = rotation
        self.scale = scale
        self.mirrored = mirrored
        self.repetition = repetition
        self.annotations = annotations if annotations is not None else {}

    def __copy__(self) -> 'Ref':
        new = Ref(
            offset=self.offset.copy(),
            rotation=self.rotation,
            scale=self.scale,
            mirrored=self.mirrored,
            repetition=copy.deepcopy(self.repetition),
            annotations=copy.deepcopy(self.annotations),
            )
        return new

    def __deepcopy__(self, memo: dict | None = None) -> 'Ref':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        #new.repetition = copy.deepcopy(self.repetition, memo)
        #new.annotations = copy.deepcopy(self.annotations, memo)
        return new

    def as_pattern(
            self,
            pattern: 'Pattern',
            ) -> 'Pattern':
        """
        Args:
            pattern: Pattern object to transform

        Returns:
            A copy of the referenced Pattern which has been scaled, rotated, etc.
             according to this `Ref`'s properties.
        """
        pattern = pattern.deepcopy()

        if self.scale != 1:
            pattern.scale_by(self.scale)
        if self.mirrored:
            pattern.mirror()
        if self.rotation % (2 * pi) != 0:
            pattern.rotate_around((0.0, 0.0), self.rotation)
        if numpy.any(self.offset):
            pattern.translate_elements(self.offset)

        if self.repetition is not None:
            combined = type(pattern)()
            for dd in self.repetition.displacements:
                temp_pat = pattern.deepcopy()
                temp_pat.ports = {}
                temp_pat.translate_elements(dd)
                combined.append(temp_pat)
            pattern = combined

        return pattern

    def rotate(self, rotation: float) -> Self:
        self.rotation += rotation
        if self.repetition is not None:
            self.repetition.rotate(rotation)
        return self

    def mirror(self, axis: int = 0) -> Self:
        self.mirror_target(axis)
        self.rotation *= -1
        if self.repetition is not None:
            self.repetition.mirror(axis)
        return self

    def mirror_target(self, axis: int = 0) -> Self:
        self.mirrored = not self.mirrored
        self.rotation += axis * pi
        return self

    def mirror2d_target(self, across_x: bool = False, across_y: bool = False) -> Self:
        self.mirrored = bool((self.mirrored + across_x + across_y) % 2)
        if across_y:
            self.rotation += pi
        return self

    def get_bounds_single(
            self,
            pattern: 'Pattern',
            *,
            library: Mapping[str, 'Pattern'] | None = None,
            ) -> NDArray[numpy.float64] | None:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the `Ref` in each dimension.
        Returns `None` if the contained `Pattern` is empty.

        Args:
            library: Name-to-Pattern mapping for resul

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        if pattern.is_empty():
            # no need to run as_pattern()
            return None

        # if rotation is manhattan, can take pattern's bounds and transform them
        if numpy.isclose(self.rotation % (pi / 2), 0):
            unrot_bounds = pattern.get_bounds(library)
            if unrot_bounds is None:
                return None

            if self.mirrored:
                unrot_bounds[:, 1] *= -1

            corners = (rotation_matrix_2d(self.rotation) @ unrot_bounds.T).T
            bounds = numpy.vstack((numpy.min(corners, axis=0),
                                   numpy.max(corners, axis=0))) * self.scale + [self.offset]
            return bounds
        return self.as_pattern(pattern=pattern).get_bounds(library)

    def __repr__(self) -> str:
        rotation = f' r{numpy.rad2deg(self.rotation):g}' if self.rotation != 0 else ''
        scale = f' d{self.scale:g}' if self.scale != 1 else ''
        mirrored = ' m' if self.mirrored else ''
        return f'<Ref {self.offset}{rotation}{scale}{mirrored}>'
