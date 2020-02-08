"""
    Repetitions provides support for efficiently nesting multiple identical
     instances of a Pattern in the same parent Pattern.
"""

from typing import Union, List, Dict, Tuple
import copy

import numpy
from numpy import pi

from .error import PatternError, PatternLockedError
from .utils import is_scalar, rotation_matrix_2d, vector2


__author__ = 'Jan Petykiewicz'


# TODO need top-level comment about what order rotation/scale/offset/mirror/array are applied

class GridRepetition:
    """
    GridRepetition provides support for efficiently embedding multiple copies of a Pattern
     into another Pattern at regularly-spaced offsets.
    """
    __slots__ = ('pattern',
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

    pattern: 'Pattern'

    _offset: numpy.ndarray
    _dose: float

    _rotation: float
    ''' Applies to individual instances in the grid, not the grid vectors '''
    _scale: float
    ''' Applies to individual instances in the grid, not the grid vectors '''
    _mirrored: List[bool]
    ''' Applies to individual instances in the grid, not the grid vectors '''

    _a_vector: numpy.ndarray
    _b_vector: numpy.ndarray or None
    _a_count: int
    _b_count: int

    identifier: Tuple
    locked: bool

    def __init__(self,
                 pattern: 'Pattern',
                 a_vector: numpy.ndarray,
                 a_count: int,
                 b_vector: numpy.ndarray = None,
                 b_count: int = 1,
                 offset: vector2 = (0.0, 0.0),
                 rotation: float = 0.0,
                 mirrored: List[bool] = None,
                 dose: float = 1.0,
                 scale: float = 1.0,
                 locked: bool = False):
        """
        :param a_vector: First lattice vector, of the form [x, y].
            Specifies center-to-center spacing between adjacent elements.
        :param a_count: Number of elements in the a_vector direction.
        :param b_vector: Second lattice vector, of the form [x, y].
            Specifies center-to-center spacing between adjacent elements.
            Can be omitted when specifying a 1D array.
        :param b_count: Number of elements in the b_vector direction.
            Should be omitted if b_vector was omitted.
        :param locked: Whether the subpattern is locked after initialization.
        :raises: InvalidDataError if b_* inputs conflict with each other
            or a_count < 1.
        """
        if b_vector is None:
            if b_count > 1:
                raise PatternError('Repetition has b_count > 1 but no b_vector')
            else:
                b_vector = numpy.array([0.0, 0.0])

        if a_count < 1:
            raise InvalidDataError('Repetition has too-small a_count: '
                                   '{}'.format(a_count))
        if b_count < 1:
            raise InvalidDataError('Repetition has too-small b_count: '
                                   '{}'.format(b_count))
        self.unlock()
        self.a_vector = a_vector
        self.b_vector = b_vector
        self.a_count = a_count
        self.b_count = b_count

        self.identifier = ()
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

    def  __deepcopy__(self, memo: Dict = None) -> 'GridReptition':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new.pattern = copy.deepcopy(self.pattern, memo)
        new.locked = self.locked
        return new

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
    def mirrored(self) -> List[bool]:
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: List[bool]):
        if is_scalar(val):
            raise PatternError('Mirrored must be a 2-element list of booleans')
        self._mirrored = numpy.array(val, dtype=bool)

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
        self._a_vector = val.flatten()

    # b_vector property
    @property
    def b_vector(self) -> numpy.ndarray:
        return self._b_vector

    @b_vector.setter
    def b_vector(self, val: vector2):
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float)

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
          etc. according to this GridRepetitions's properties.
        :return: Copy of self.pattern that has been repeated / altered as implied by
          this object's other properties.
        """
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

        :param offset: Translate by this offset
        :return: self
        """
        self.offset += offset
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'GridRepetition':
        """
        Rotate the array around a point

        :param pivot: Point to rotate around
        :param rotation: Angle to rotate by (counterclockwise, radians)
        :return: self
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

        :param rotation: Angle to rotate by (counterclockwise, radians)
        :return: self
        """
        self.rotate_elements(rotation)
        self.a_vector = numpy.dot(rotation_matrix_2d(rotation), self.a_vector)
        if self.b_vector is not None:
            self.b_vector = numpy.dot(rotation_matrix_2d(rotation), self.b_vector)
        return self

    def rotate_elements(self, rotation: float) -> 'GridRepetition':
        """
        Rotate each element around its origin

        :param rotation: Angle to rotate by (counterclockwise, radians)
        :return: self
        """
        self.rotation += rotation
        return self

    def mirror(self, axis: int) -> 'GridRepetition':
        """
        Mirror the GridRepetition across an axis.

        :param axis: Axis to mirror across.
        :return: self
        """
        self.mirror_elements(axis)
        self.a_vector[axis] *= -1
        if self.b_vector is not None:
            self.b_vector[axis] *= -1
        return self

    def mirror_elements(self, axis: int) -> 'GridRepetition':
        """
        Mirror each element across an axis relative to its origin.

        :param axis: Axis to mirror across.
        :return: self
        """
        self.mirrored[axis] = not self.mirrored[axis]
        self.rotation *= -1
        return self

    def get_bounds(self) -> numpy.ndarray or None:
        """
        Return a numpy.ndarray containing [[x_min, y_min], [x_max, y_max]], corresponding to the
         extent of the GridRepetition in each dimension.
        Returns None if the contained Pattern is empty.

        :return: [[x_min, y_min], [x_max, y_max]] or None
        """
        return self.as_pattern().get_bounds()

    def scale_by(self, c: float) -> 'GridRepetition':
        """
        Scale the GridRepetition by a factor

        :param c: scaling factor
        """
        self.scale_elements_by(c)
        self.a_vector *= c
        if self.b_vector is not None:
            self.b_vector *= c
        return self

    def scale_elements_by(self, c: float) -> 'GridRepetition':
        """
        Scale each element by a factor

        :param c: scaling factor
        """
        self.scale *= c
        return self

    def copy(self) -> 'GridRepetition':
        """
        Return a shallow copy of the repetition.

        :return: copy.copy(self)
        """
        return copy.copy(self)

    def deepcopy(self) -> 'GridRepetition':
        """
        Return a deep copy of the repetition.

        :return: copy.copy(self)
        """
        return copy.deepcopy(self)

    def lock(self) -> 'GridRepetition':
        """
        Lock the GridRepetition

        :return: self
        """
        object.__setattr__(self, 'locked', True)
        return self

    def unlock(self) -> 'GridRepetition':
        """
        Unlock the GridRepetition

        :return: self
        """
        object.__setattr__(self, 'locked', False)
        return self

    def deeplock(self) -> 'GridRepetition':
        """
        Recursively lock the GridRepetition and its contained pattern

        :return: self
        """
        self.lock()
        self.pattern.deeplock()
        return self

    def deepunlock(self) -> 'GridRepetition':
        """
        Recursively unlock the GridRepetition and its contained pattern

        This is dangerous unless you have just performed a deepcopy!

        :return: self
        """
        self.unlock()
        self.pattern.deepunlock()
        return self
