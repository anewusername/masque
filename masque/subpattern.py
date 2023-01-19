"""
 SubPattern provides basic support for nesting Pattern objects within each other, by adding
  offset, rotation, scaling, and other such properties to the reference.
"""
#TODO more top-level documentation

from typing import Dict, Tuple, Optional, Sequence, Mapping, TYPE_CHECKING, Any, TypeVar
import copy

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from .error import PatternError
from .utils import is_scalar, AutoSlots, annotations_t
from .repetition import Repetition
from .traits import (
    PositionableImpl, DoseableImpl, RotatableImpl, ScalableImpl,
    Mirrorable, PivotableImpl, Copyable, RepeatableImpl, AnnotatableImpl,
    )


if TYPE_CHECKING:
    from . import Pattern


S = TypeVar('S', bound='SubPattern')


class SubPattern(PositionableImpl, DoseableImpl, RotatableImpl, ScalableImpl, Mirrorable,
                 PivotableImpl, Copyable, RepeatableImpl, AnnotatableImpl,
                 metaclass=AutoSlots):
    """
    SubPattern provides basic support for nesting Pattern objects within each other, by adding
     offset, rotation, scaling, and associated methods.
    """
    __slots__ = ('_target', '_mirrored')

    _target: Optional[str]
    """ The name of the `Pattern` being instanced """

    _mirrored: NDArray[numpy.bool_]
    """ Whether to mirror the instance across the x and/or y axes. """

    def __init__(
            self,
            target: Optional[str],
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            mirrored: Optional[Sequence[bool]] = None,
            dose: float = 1.0,
            scale: float = 1.0,
            repetition: Optional[Repetition] = None,
            annotations: Optional[annotations_t] = None,
            ) -> None:
        """
        Args:
            target: Name of the Pattern to reference.
            offset: (x, y) offset applied to the referenced pattern. Not affected by rotation etc.
            rotation: Rotation (radians, counterclockwise) relative to the referenced pattern's (0, 0).
            mirrored: Whether to mirror the referenced pattern across its x and y axes.
            dose: Scaling factor applied to the dose.
            scale: Scaling factor applied to the pattern's geometry.
            repetition: `Repetition` object, default `None`
        """
        self.target = target
        self.offset = offset
        self.rotation = rotation
        self.dose = dose
        self.scale = scale
        if mirrored is None:
            mirrored = (False, False)
        self.mirrored = mirrored
        self.repetition = repetition
        self.annotations = annotations if annotations is not None else {}

    def __copy__(self) -> 'SubPattern':
        new = SubPattern(
            target=self.target,
            offset=self.offset.copy(),
            rotation=self.rotation,
            dose=self.dose,
            scale=self.scale,
            mirrored=self.mirrored.copy(),
            repetition=copy.deepcopy(self.repetition),
            annotations=copy.deepcopy(self.annotations),
            )
        return new

    def __deepcopy__(self, memo: Optional[Dict] = None) -> 'SubPattern':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new.repetition = copy.deepcopy(self.repetition, memo)
        new.annotations = copy.deepcopy(self.annotations, memo)
        return new

    # target property
    @property
    def target(self) -> Optional[str]:
        return self._target

    @target.setter
    def target(self, val: Optional[str]) -> None:
        if val is not None and not isinstance(val, str):
            raise PatternError(f'Provided target {val} is not a str or None!')
        self._target = val

    # Mirrored property
    @property
    def mirrored(self) -> Any:   #TODO mypy#3004  NDArray[numpy.bool_]:
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: ArrayLike) -> None:
        if is_scalar(val):
            raise PatternError('Mirrored must be a 2-element list of booleans')
        self._mirrored = numpy.array(val, dtype=bool, copy=True)

    def as_pattern(
            self,
            *,
            pattern: Optional[Pattern] = None,
            library: Optional[Mapping[str, Pattern]] = None,
            ) -> 'Pattern':
        """
        Args:
            pattern: Pattern object to transform
            library: A str->Pattern mapping, used instead of `pattern`. Must contain
                `self.target`.

        Returns:
            A copy of the referenced Pattern which has been scaled, rotated, etc.
             according to this `SubPattern`'s properties.
        """
        if pattern is None:
            if library is None:
                raise PatternError('as_pattern() must be given a pattern or library.')

            assert(self.target is not None)
            pattern = library[self.target]

        pattern = pattern.deepcopy()

        if self.scale != 1:
            pattern.scale_by(self.scale)
        if numpy.any(self.mirrored):
            pattern.mirror2d(self.mirrored)
        if self.rotation % (2 * pi) != 0:
            pattern.rotate_around((0.0, 0.0), self.rotation)
        if numpy.any(self.offset):
            pattern.translate_elements(self.offset)
        if self.dose != 1:
            pattern.scale_element_doses(self.dose)

        if self.repetition is not None:
            combined = type(pattern)()
            for dd in self.repetition.displacements:
                temp_pat = pattern.deepcopy()
                temp_pat.translate_elements(dd)
                combined.append(temp_pat)
            pattern = combined

        return pattern

    def rotate(self: S, rotation: float) -> S:
        self.rotation += rotation
        if self.repetition is not None:
            self.repetition.rotate(rotation)
        return self

    def mirror(self: S, axis: int) -> S:
        self.mirrored[axis] = not self.mirrored[axis]
        self.rotation *= -1
        if self.repetition is not None:
            self.repetition.mirror(axis)
        return self

    def get_bounds(
            self,
            *,
            pattern: Optional[Pattern] = None,
            library: Optional[Mapping[str, Pattern]] = None,
            ) -> Optional[NDArray[numpy.float64]]:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the `SubPattern` in each dimension.
        Returns `None` if the contained `Pattern` is empty.

        Args:
            library: Name-to-Pattern mapping for resul

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        if pattern is None and library is None:
            raise PatternError('as_pattern() must be given a pattern or library.')
        if pattern is None and self.target is None:
            return None
        return self.as_pattern(pattern=pattern, library=library).get_bounds()

    def __repr__(self) -> str:
        name = f'"{self.target}"' if self.target is not None else None
        rotation = f' r{self.rotation*180/pi:g}' if self.rotation != 0 else ''
        scale = f' d{self.scale:g}' if self.scale != 1 else ''
        mirrored = ' m{:d}{:d}'.format(*self.mirrored) if self.mirrored.any() else ''
        dose = f' d{self.dose:g}' if self.dose != 1 else ''
        return f'<SubPattern {name} at {self.offset}{rotation}{scale}{mirrored}{dose}>'
