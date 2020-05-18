"""
 SubPattern provides basic support for nesting Pattern objects within each other, by adding
  offset, rotation, scaling, and other such properties to the reference.
"""
#TODO more top-level documentation

from typing import Union, List, Dict, Tuple, Optional, Sequence, TYPE_CHECKING, Any
import copy

import numpy
from numpy import pi

from .error import PatternError, PatternLockedError
from .utils import is_scalar, rotation_matrix_2d, vector2
from .repetition import GridRepetition


if TYPE_CHECKING:
    from . import Pattern


class SubPattern:
    """
    SubPattern provides basic support for nesting Pattern objects within each other, by adding
     offset, rotation, scaling, and associated methods.
    """
    __slots__ = ('_pattern',
                 '_offset',
                 '_rotation',
                 '_dose',
                 '_scale',
                 '_mirrored',
                 'identifier',
                 'locked')

    _pattern: Optional['Pattern']
    """ The `Pattern` being instanced """

    _offset: numpy.ndarray
    """ (x, y) offset for the instance """

    _rotation: float
    """ rotation for the instance, radians counterclockwise """

    _dose: float
    """ dose factor for the instance """

    _scale: float
    """ scale factor for the instance """

    _mirrored: numpy.ndarray        # ndarray[bool]
    """ Whether to mirror the instanc across the x and/or y axes. """

    identifier: Tuple[Any, ...]
    """ Arbitrary identifier, used internally by some `masque` functions. """

    locked: bool
    """ If `True`, disallows changes to the GridRepetition """

    def __init__(self,
                 pattern: Optional['Pattern'],
                 offset: vector2 = (0.0, 0.0),
                 rotation: float = 0.0,
                 mirrored: Optional[Sequence[bool]] = None,
                 dose: float = 1.0,
                 scale: float = 1.0,
                 locked: bool = False,
                 identifier: Tuple[Any, ...] = ()):
        """
        Args:
            pattern: Pattern to reference.
            offset: (x, y) offset applied to the referenced pattern. Not affected by rotation etc.
            rotation: Rotation (radians, counterclockwise) relative to the referenced pattern's (0, 0).
            mirrored: Whether to mirror the referenced pattern across its x and y axes.
            dose: Scaling factor applied to the dose.
            scale: Scaling factor applied to the pattern's geometry.
            locked: Whether the `SubPattern` is locked after initialization.
            identifier: Arbitrary tuple, used internally by some `masque` functions.
        """
        object.__setattr__(self, 'locked', False)
        self.identifier = identifier
        self.pattern = pattern
        self.offset = offset
        self.rotation = rotation
        self.dose = dose
        self.scale = scale
        if mirrored is None:
            mirrored = [False, False]
        self.mirrored = mirrored
        self.locked = locked

    def __setattr__(self, name, value):
        if self.locked and name != 'locked':
            raise PatternLockedError()
        object.__setattr__(self, name, value)

    def  __copy__(self) -> 'SubPattern':
        new = SubPattern(pattern=self.pattern,
                         offset=self.offset.copy(),
                         rotation=self.rotation,
                         dose=self.dose,
                         scale=self.scale,
                         mirrored=self.mirrored.copy(),
                         locked=self.locked)
        return new

    def  __deepcopy__(self, memo: Dict = None) -> 'SubPattern':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new.pattern = copy.deepcopy(self.pattern, memo)
        new.locked = self.locked
        return new

    # pattern property
    @property
    def pattern(self) -> Optional['Pattern']:
        return self._pattern

    @pattern.setter
    def pattern(self, val: Optional['Pattern']):
        from .pattern import Pattern
        if val is not None and not isinstance(val, Pattern):
            raise PatternError('Provided pattern {} is not a Pattern object or None!'.format(val))
        self._pattern = val

    # offset property
    @property
    def offset(self) -> numpy.ndarray:
        return self._offset

    @offset.setter
    def offset(self, val: vector2):
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float)

        if val.size != 2:
            raise PatternError('Offset must be convertible to size-2 ndarray')
        self._offset = val.flatten().astype(float)

    # dose property
    @property
    def dose(self) -> float:
        return self._dose

    @dose.setter
    def dose(self, val: float):
        if not is_scalar(val):
            raise PatternError('Dose must be a scalar')
        if not val >= 0:
            raise PatternError('Dose must be non-negative')
        self._dose = val

    # scale property
    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, val: float):
        if not is_scalar(val):
            raise PatternError('Scale must be a scalar')
        if not val > 0:
            raise PatternError('Scale must be positive')
        self._scale = val

    # Rotation property [ccw]
    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, val: float):
        if not is_scalar(val):
            raise PatternError('Rotation must be a scalar')
        self._rotation = val % (2 * pi)

    # Mirrored property
    @property
    def mirrored(self) -> numpy.ndarray:        # ndarray[bool]
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: Sequence[bool]):
        if is_scalar(val):
            raise PatternError('Mirrored must be a 2-element list of booleans')
        self._mirrored = numpy.array(val, dtype=bool, copy=True)

    def as_pattern(self) -> 'Pattern':
        """
        Returns:
            A copy of self.pattern which has been scaled, rotated, etc. according to this
             `SubPattern`'s properties.
        """
        assert(self.pattern is not None)
        pattern = self.pattern.deepcopy().deepunlock()
        pattern.scale_by(self.scale)
        [pattern.mirror(ax) for ax, do in enumerate(self.mirrored) if do]
        pattern.rotate_around((0.0, 0.0), self.rotation)
        pattern.translate_elements(self.offset)
        pattern.scale_element_doses(self.dose)
        return pattern

    def translate(self, offset: vector2) -> 'SubPattern':
        """
        Translate by the given offset

        Args:
            offset: Offset `[x, y]` to translate by

        Returns:
            self
        """
        self.offset += offset
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'SubPattern':
        """
        Rotate around a point

        Args:
            pivot: Point `[x, y]` to rotate around
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        pivot = numpy.array(pivot, dtype=float)
        self.translate(-pivot)
        self.offset = numpy.dot(rotation_matrix_2d(rotation), self.offset)
        self.rotate(rotation)
        self.translate(+pivot)
        return self

    def rotate(self, rotation: float) -> 'SubPattern':
        """
        Rotate the instance around it's origin

        Args:
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        self.rotation += rotation
        return self

    def mirror(self, axis: int) -> 'SubPattern':
        """
        Mirror the subpattern across an axis.

        Args:
            axis: Axis to mirror across.

        Returns:
            self
        """
        self.mirrored[axis] = not self.mirrored[axis]
        self.rotation *= -1
        return self

    def get_bounds(self) -> Optional[numpy.ndarray]:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the `SubPattern` in each dimension.
        Returns `None` if the contained `Pattern` is empty.

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        if self.pattern is None:
            return None
        return self.as_pattern().get_bounds()

    def scale_by(self, c: float) -> 'SubPattern':
        """
        Scale the subpattern by a factor

        Args:
            c: scaling factor

        Returns:
            self
        """
        self.scale *= c
        return self

    def copy(self) -> 'SubPattern':
        """
        Return a shallow copy of the subpattern.

        Returns:
            `copy.copy(self)`
        """
        return copy.copy(self)

    def deepcopy(self) -> 'SubPattern':
        """
        Return a deep copy of the subpattern.

        Returns:
            `copy.deepcopy(self)`
        """
        return copy.deepcopy(self)

    def lock(self) -> 'SubPattern':
        """
        Lock the SubPattern, disallowing changes

        Returns:
            self
        """
        self.offset.flags.writeable = False
        self.mirrored.flags.writeable = False
        object.__setattr__(self, 'locked', True)
        return self

    def unlock(self) -> 'SubPattern':
        """
        Unlock the SubPattern

        Returns:
            self
        """
        self.offset.flags.writeable = True
        self.mirrored.flags.writeable = True
        object.__setattr__(self, 'locked', False)
        return self

    def deeplock(self) -> 'SubPattern':
        """
        Recursively lock the SubPattern and its contained pattern

        Returns:
            self
        """
        assert(self.pattern is not None)
        self.lock()
        self.pattern.deeplock()
        return self

    def deepunlock(self) -> 'SubPattern':
        """
        Recursively unlock the SubPattern and its contained pattern

        This is dangerous unless you have just performed a deepcopy, since
        the subpattern and its components may be used in more than one once!

        Returns:
            self
        """
        assert(self.pattern is not None)
        self.unlock()
        self.pattern.deepunlock()
        return self

    def __repr__(self) -> str:
        name = self.pattern.name if self.pattern is not None else None
        rotation = f' r{self.rotation*180/pi:g}' if self.rotation != 0 else ''
        scale = f' d{self.scale:g}' if self.scale != 1 else ''
        mirrored = ' m{:d}{:d}'.format(*self.mirrored) if self.mirrored.any() else ''
        dose = f' d{self.dose:g}' if self.dose != 1 else ''
        locked = ' L' if self.locked else ''
        return f'<SubPattern "{name}" at {self.offset}{rotation}{scale}{mirrored}{dose}{locked}>'


subpattern_t = Union[SubPattern, GridRepetition]
