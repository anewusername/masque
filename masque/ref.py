"""
 Ref provides basic support for nesting Pattern objects within each other, by adding
  offset, rotation, scaling, and other such properties to the reference.
"""
#TODO more top-level documentation

from typing import Sequence, Mapping, TYPE_CHECKING, Any, Self
import copy

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from .error import PatternError
from .utils import is_scalar, annotations_t
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
    `Ref` provides basic support for nesting Pattern objects within each other, by adding
     offset, rotation, scaling, and associated methods.
    """
    __slots__ = (
        '_mirrored',
        # inherited
        '_offset', '_rotation', 'scale', '_repetition', '_annotations',
        )

    _mirrored: NDArray[numpy.bool_]
    """ Whether to mirror the instance across the x and/or y axes. """

    def __init__(
            self,
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            mirrored: Sequence[bool] | None = None,
            scale: float = 1.0,
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            ) -> None:
        """
        Args:
            offset: (x, y) offset applied to the referenced pattern. Not affected by rotation etc.
            rotation: Rotation (radians, counterclockwise) relative to the referenced pattern's (0, 0).
            mirrored: Whether to mirror the referenced pattern across its x and y axes.
            scale: Scaling factor applied to the pattern's geometry.
            repetition: `Repetition` object, default `None`
        """
        self.offset = offset
        self.rotation = rotation
        self.scale = scale
        if mirrored is None:
            mirrored = (False, False)
        self.mirrored = mirrored
        self.repetition = repetition
        self.annotations = annotations if annotations is not None else {}

    def __copy__(self) -> 'Ref':
        new = Ref(
            offset=self.offset.copy(),
            rotation=self.rotation,
            scale=self.scale,
            mirrored=self.mirrored.copy(),
            repetition=copy.deepcopy(self.repetition),
            annotations=copy.deepcopy(self.annotations),
            )
        return new

    def __deepcopy__(self, memo: dict | None = None) -> 'Ref':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new.repetition = copy.deepcopy(self.repetition, memo)
        new.annotations = copy.deepcopy(self.annotations, memo)
        return new

    # Mirrored property
    @property
    def mirrored(self) -> Any:   # TODO mypy#3004  NDArray[numpy.bool_]:
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: ArrayLike) -> None:
        if is_scalar(val):
            raise PatternError('Mirrored must be a 2-element list of booleans')
        self._mirrored = numpy.array(val, dtype=bool, copy=True)

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
        if numpy.any(self.mirrored):
            pattern.mirror2d(self.mirrored)
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

    def mirror(self, axis: int) -> Self:
        self.mirrored[axis] = not self.mirrored[axis]
        self.rotation *= -1
        if self.repetition is not None:
            self.repetition.mirror(axis)
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
        return self.as_pattern(pattern=pattern).get_bounds(library)     # TODO can just take pattern's bounds and then transform those!

    def __repr__(self) -> str:
        rotation = f' r{numpy.rad2deg(self.rotation):g}' if self.rotation != 0 else ''
        scale = f' d{self.scale:g}' if self.scale != 1 else ''
        mirrored = ' m{:d}{:d}'.format(*self.mirrored) if self.mirrored.any() else ''
        return f'<Ref {self.offset}{rotation}{scale}{mirrored}>'
