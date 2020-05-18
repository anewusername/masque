"""
    Repetitions provides support for efficiently nesting multiple identical
     instances of a Pattern in the same parent Pattern.
"""

from typing import Union, List, Dict, Tuple, Optional, Sequence, TYPE_CHECKING, Any
import copy

import numpy
from numpy import pi

from .error import PatternError, PatternLockedError
from .utils import is_scalar, rotation_matrix_2d, vector2

if TYPE_CHECKING:
    from . import Pattern


# TODO need top-level comment about what order rotation/scale/offset/mirror/array are applied

class GridRepetition:
    """
    GridRepetition provides support for efficiently embedding multiple copies of a `Pattern`
     into another `Pattern` at regularly-spaced offsets.

    Note that rotation, scaling, and mirroring are applied to individual instances of the
     pattern, not to the grid vectors.

    The order of operations is
        1. A single refernce instance to the target pattern is mirrored
        2. The single instance is rotated.
        3. The instance is scaled by the scaling factor.
        4. The instance is shifted by the provided offset
            (no mirroring/scaling/rotation is applied to the offset).
        5. Additional copies of the instance will appear at coordinates specified by
            `(offset + aa * a_vector + bb * b_vector)`, with `aa in range(0, a_count)`
            and `bb in range(0, b_count)`. All instance locations remain unaffected by
            mirroring/scaling/rotation, though each instance's data will be transformed
            relative to the instance's location (i.e. relative to the contained pattern's
            (0, 0) point).
    """
    __slots__ = ('_pattern',
                 '_offset',
                 '_rotation',
                 '_dose',
                 '_scale',
                 '_mirrored',
                 '_a_vector',
                 '_b_vector',
                 '_a_count',
                 '_b_count',
                 'identifier',
                 'locked')

    _pattern: Optional['Pattern']
    """ The `Pattern` being instanced """

    _offset: numpy.ndarray
    """ (x, y) offset for the base instance """

    _dose: float
    """ Scaling factor applied to the dose """

    _rotation: float
    """ Rotation of the individual instances in the grid (not the grid vectors).
    Radians, counterclockwise.
    """

    _scale: float
    """ Scaling factor applied to individual instances in the grid (not the grid vectors) """

    _mirrored: numpy.ndarray        # ndarray[bool]
    """ Whether to mirror individual instances across the x and y axes
    (Applies to individual instances in the grid, not the grid vectors)
    """

    _a_vector: numpy.ndarray
    """ Vector `[x, y]` specifying the first lattice vector of the grid.
        Specifies center-to-center spacing between adjacent elements.
    """

    _a_count: int
    """ Number of instances along the direction specified by the `a_vector` """

    _b_vector: Optional[numpy.ndarray]
    """ Vector `[x, y]` specifying a second lattice vector for the grid.
        Specifies center-to-center spacing between adjacent elements.
        Can be `None` for a 1D array.
    """

    _b_count: int
    """ Number of instances along the direction specified by the `b_vector` """

    identifier: Tuple[Any, ...]
    """ Arbitrary identifier, used internally by some `masque` functions. """

    locked: bool
    """ If `True`, disallows changes to the GridRepetition """

    def __init__(self,
                 pattern: Optional['Pattern'],
                 a_vector: numpy.ndarray,
                 a_count: int,
                 b_vector: Optional[numpy.ndarray] = None,
                 b_count: Optional[int] = 1,
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
            a_vector: First lattice vector, of the form `[x, y]`.
                Specifies center-to-center spacing between adjacent instances.
            a_count: Number of elements in the a_vector direction.
            b_vector: Second lattice vector, of the form `[x, y]`.
                Specifies center-to-center spacing between adjacent instances.
                Can be omitted when specifying a 1D array.
            b_count: Number of elements in the `b_vector` direction.
                Should be omitted if `b_vector` was omitted.
            offset: (x, y) offset applied to all instances.
            rotation: Rotation (radians, counterclockwise) applied to each instance.
                       Relative to each instance's (0, 0).
            mirrored: Whether to mirror individual instances across the x and y axes.
            dose: Scaling factor applied to the dose.
            scale: Scaling factor applied to the instances' geometry.
            locked: Whether the `GridRepetition` is locked after initialization.
            identifier: Arbitrary tuple, used internally by some `masque` functions.

        Raises:
            PatternError if `b_*` inputs conflict with each other
            or `a_count < 1`.
        """
        if b_count is None:
            b_count = 1

        if b_vector is None:
            if b_count > 1:
                raise PatternError('Repetition has b_count > 1 but no b_vector')
            else:
                b_vector = numpy.array([0.0, 0.0])

        if a_count < 1:
            raise PatternError('Repetition has too-small a_count: '
                               '{}'.format(a_count))
        if b_count < 1:
            raise PatternError('Repetition has too-small b_count: '
                               '{}'.format(b_count))

        object.__setattr__(self, 'locked', False)
        self.a_vector = a_vector
        self.b_vector = b_vector
        self.a_count = a_count
        self.b_count = b_count

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

    def  __copy__(self) -> 'GridRepetition':
        new = GridRepetition(pattern=self.pattern,
                             a_vector=self.a_vector.copy(),
                             b_vector=copy.copy(self.b_vector),
                             a_count=self.a_count,
                             b_count=self.b_count,
                             offset=self.offset.copy(),
                             rotation=self.rotation,
                             dose=self.dose,
                             scale=self.scale,
                             mirrored=self.mirrored.copy(),
                             locked=self.locked)
        return new

    def  __deepcopy__(self, memo: Dict = None) -> 'GridRepetition':
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
        if self.locked:
            raise PatternLockedError()

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

    # a_vector property
    @property
    def a_vector(self) -> numpy.ndarray:
        return self._a_vector

    @a_vector.setter
    def a_vector(self, val: vector2):
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float)

        if val.size != 2:
            raise PatternError('a_vector must be convertible to size-2 ndarray')
        self._a_vector = val.flatten().astype(float)

    # b_vector property
    @property
    def b_vector(self) -> numpy.ndarray:
        return self._b_vector

    @b_vector.setter
    def b_vector(self, val: vector2):
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float, copy=True)

        if val.size != 2:
            raise PatternError('b_vector must be convertible to size-2 ndarray')
        self._b_vector = val.flatten()

    # a_count property
    @property
    def a_count(self) -> int:
        return self._a_count

    @a_count.setter
    def a_count(self, val: int):
        if val != int(val):
            raise PatternError('a_count must be convertable to an int!')
        self._a_count = int(val)

    # b_count property
    @property
    def b_count(self) -> int:
        return self._b_count

    @b_count.setter
    def b_count(self, val: int):
        if val != int(val):
            raise PatternError('b_count must be convertable to an int!')
        self._b_count = int(val)

    def as_pattern(self) -> 'Pattern':
        """
        Returns a copy of self.pattern which has been scaled, rotated, repeated, etc.
          etc. according to this `GridRepetition`'s properties.

        Returns:
            A copy of self.pattern which has been scaled, rotated, repeated, etc.
             etc. according to this `GridRepetition`'s properties.
        """
        assert(self.pattern is not None)
        patterns = []

        for a in range(self.a_count):
            for b in range(self.b_count):
                offset = a * self.a_vector + b * self.b_vector
                newPat = self.pattern.deepcopy().deepunlock()
                newPat.translate_elements(offset)
                patterns.append(newPat)

        combined = patterns[0]
        for p in patterns[1:]:
            combined.append(p)

        combined.scale_by(self.scale)
        [combined.mirror(ax) for ax, do in enumerate(self.mirrored) if do]
        combined.rotate_around((0.0, 0.0), self.rotation)
        combined.translate_elements(self.offset)
        combined.scale_element_doses(self.dose)

        return combined

    def translate(self, offset: vector2) -> 'GridRepetition':
        """
        Translate by the given offset

        Args:
            offset: `[x, y]` to translate by

        Returns:
            self
        """
        self.offset += offset
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'GridRepetition':
        """
        Rotate the array around a point

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

    def rotate(self, rotation: float) -> 'GridRepetition':
        """
        Rotate around (0, 0)

        Args:
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        self.rotate_elements(rotation)
        self.a_vector = numpy.dot(rotation_matrix_2d(rotation), self.a_vector)
        if self.b_vector is not None:
            self.b_vector = numpy.dot(rotation_matrix_2d(rotation), self.b_vector)
        return self

    def rotate_elements(self, rotation: float) -> 'GridRepetition':
        """
        Rotate each element around its origin

        Args:
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        self.rotation += rotation
        return self

    def mirror(self, axis: int) -> 'GridRepetition':
        """
        Mirror the GridRepetition across an axis.

        Args:
            axis: Axis to mirror across.
                (0: mirror across x-axis, 1: mirror across y-axis)

        Returns:
            self
        """
        self.mirror_elements(axis)
        self.a_vector[1-axis] *= -1
        if self.b_vector is not None:
            self.b_vector[1-axis] *= -1
        return self

    def mirror_elements(self, axis: int) -> 'GridRepetition':
        """
        Mirror each element across an axis relative to its origin.

        Args:
            axis: Axis to mirror across.
                (0: mirror across x-axis, 1: mirror across y-axis)

        Returns:
            self
        """
        self.mirrored[axis] = not self.mirrored[axis]
        self.rotation *= -1
        return self

    def get_bounds(self) -> Optional[numpy.ndarray]:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the `GridRepetition` in each dimension.
        Returns `None` if the contained `Pattern` is empty.

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        if self.pattern is None:
            return None
        return self.as_pattern().get_bounds()

    def scale_by(self, c: float) -> 'GridRepetition':
        """
        Scale the GridRepetition by a factor

        Args:
            c: scaling factor

        Returns:
            self
        """
        self.scale_elements_by(c)
        self.a_vector *= c
        if self.b_vector is not None:
            self.b_vector *= c
        return self

    def scale_elements_by(self, c: float) -> 'GridRepetition':
        """
        Scale each element by a factor

        Args:
            c: scaling factor

        Returns:
            self
        """
        self.scale *= c
        return self

    def copy(self) -> 'GridRepetition':
        """
        Return a shallow copy of the repetition.

        Returns:
            `copy.copy(self)`
        """
        return copy.copy(self)

    def deepcopy(self) -> 'GridRepetition':
        """
        Return a deep copy of the repetition.

        Returns:
            `copy.deepcopy(self)`
        """
        return copy.deepcopy(self)

    def lock(self) -> 'GridRepetition':
        """
        Lock the `GridRepetition`, disallowing changes.

        Returns:
            self
        """
        self.offset.flags.writeable = False
        self.a_vector.flags.writeable = False
        self.mirrored.flags.writeable = False
        if self.b_vector is not None:
            self.b_vector.flags.writeable = False
        object.__setattr__(self, 'locked', True)
        return self

    def unlock(self) -> 'GridRepetition':
        """
        Unlock the `GridRepetition`

        Returns:
            self
        """
        self.offset.flags.writeable = True
        self.a_vector.flags.writeable = True
        self.mirrored.flags.writeable = True
        if self.b_vector is not None:
            self.b_vector.flags.writeable = True
        object.__setattr__(self, 'locked', False)
        return self

    def deeplock(self) -> 'GridRepetition':
        """
        Recursively lock the `GridRepetition` and its contained pattern

        Returns:
            self
        """
        assert(self.pattern is not None)
        self.lock()
        self.pattern.deeplock()
        return self

    def deepunlock(self) -> 'GridRepetition':
        """
        Recursively unlock the `GridRepetition` and its contained pattern

        This is dangerous unless you have just performed a deepcopy, since
            the component parts may be reused elsewhere.

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
        bv = f', {self.b_vector}' if self.b_vector is not None else ''
        return (f'<GridRepetition "{name}" at {self.offset} {rotation}{scale}{mirrored}{dose}'
                f' {self.a_count}x{self.b_count} ({self.a_vector}{bv}){locked}>')
