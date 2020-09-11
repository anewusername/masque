"""
 SubPattern provides basic support for nesting Pattern objects within each other, by adding
  offset, rotation, scaling, and other such properties to the reference.
"""
#TODO more top-level documentation

from typing import Union, List, Dict, Tuple, Optional, Sequence, TYPE_CHECKING, Any
import copy

import numpy        # type: ignore
from numpy import pi

from .error import PatternError, PatternLockedError
from .utils import is_scalar, rotation_matrix_2d, vector2, AutoSlots, annotations_t
from .repetition import Repetition
from .traits import (PositionableImpl, DoseableImpl, RotatableImpl, ScalableImpl,
                     Mirrorable, PivotableImpl, Copyable, LockableImpl, RepeatableImpl,
                     AnnotatableImpl)


if TYPE_CHECKING:
    from . import Pattern


class SubPattern(PositionableImpl, DoseableImpl, RotatableImpl, ScalableImpl, Mirrorable,
                 PivotableImpl, Copyable, RepeatableImpl, LockableImpl, AnnotatableImpl,
                 metaclass=AutoSlots):
    """
    SubPattern provides basic support for nesting Pattern objects within each other, by adding
     offset, rotation, scaling, and associated methods.
    """
    __slots__ = ('_pattern',
                 '_mirrored',
                 'identifier',
                 )

    _pattern: Optional['Pattern']
    """ The `Pattern` being instanced """

    _mirrored: numpy.ndarray        # ndarray[bool]
    """ Whether to mirror the instance across the x and/or y axes. """

    identifier: Tuple[Any, ...]
    """ Arbitrary identifier, used internally by some `masque` functions. """

    def __init__(self,
                 pattern: Optional['Pattern'],
                 *,
                 offset: vector2 = (0.0, 0.0),
                 rotation: float = 0.0,
                 mirrored: Optional[Sequence[bool]] = None,
                 dose: float = 1.0,
                 scale: float = 1.0,
                 repetition: Optional[Repetition] = None,
                 annotations: Optional[annotations_t] = None,
                 locked: bool = False,
                 identifier: Tuple[Any, ...] = (),
                 ):
        """
        Args:
            pattern: Pattern to reference.
            offset: (x, y) offset applied to the referenced pattern. Not affected by rotation etc.
            rotation: Rotation (radians, counterclockwise) relative to the referenced pattern's (0, 0).
            mirrored: Whether to mirror the referenced pattern across its x and y axes.
            dose: Scaling factor applied to the dose.
            scale: Scaling factor applied to the pattern's geometry.
            repetition: TODO
            locked: Whether the `SubPattern` is locked after initialization.
            identifier: Arbitrary tuple, used internally by some `masque` functions.
        """
        LockableImpl.unlock(self)
        self.identifier = identifier
        self.pattern = pattern
        self.offset = offset
        self.rotation = rotation
        self.dose = dose
        self.scale = scale
        if mirrored is None:
            mirrored = [False, False]
        self.mirrored = mirrored
        self.repetition = repetition
        self.annotations = annotations if annotations is not None else {}
        self.set_locked(locked)

    def  __copy__(self) -> 'SubPattern':
        new = SubPattern(pattern=self.pattern,
                         offset=self.offset.copy(),
                         rotation=self.rotation,
                         dose=self.dose,
                         scale=self.scale,
                         mirrored=self.mirrored.copy(),
                         repetition=copy.deepcopy(self.repetition),
                         annotations=copy.deepcopy(self.annotations),
                         locked=self.locked)
        return new

    def  __deepcopy__(self, memo: Dict = None) -> 'SubPattern':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new.pattern = copy.deepcopy(self.pattern, memo)
        new.repetition = copy.deepcopy(self.repetition, memo)
        new.annotations = copy.deepcopy(self.annotations, memo)
        new.set_locked(self.locked)
        return new

    # pattern property
    @property
    def pattern(self) -> Optional['Pattern']:
        return self._pattern

    @pattern.setter
    def pattern(self, val: Optional['Pattern']):
        from .pattern import Pattern
        if val is not None and not isinstance(val, Pattern):
            raise PatternError(f'Provided pattern {val} is not a Pattern object or None!')
        self._pattern = val

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

        if self.repetition is not None:
            combined = type(pattern)(name='__repetition__')
            for dd in self.repetition.displacements:
                temp_pat = pattern.deepcopy()
                temp_pat.translate_elements(dd)
                combined.append(temp_pat)
            pattern = combined

        return pattern

    def rotate(self, rotation: float) -> 'SubPattern':
        self.rotation += rotation
        if self.repetition is not None:
            self.repetition.rotate(rotation)
        return self

    def mirror(self, axis: int) -> 'SubPattern':
        self.mirrored[axis] = not self.mirrored[axis]
        self.rotation *= -1
        if self.repetition is not None:
            self.repetition.mirror(axis)
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

    def lock(self) -> 'SubPattern':
        """
        Lock the SubPattern, disallowing changes

        Returns:
            self
        """
        self.mirrored.flags.writeable = False
        PositionableImpl._lock(self)
        LockableImpl.lock(self)
        return self

    def unlock(self) -> 'SubPattern':
        """
        Unlock the SubPattern

        Returns:
            self
        """
        LockableImpl.unlock(self)
        PositionableImpl._unlock(self)
        self.mirrored.flags.writeable = True
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
