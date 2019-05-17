"""
    Repetitions provides support for efficiently nesting multiple identical
     instances of a Pattern in the same parent Pattern.
"""

from typing import Union, List, Dict, Tuple
import copy

import numpy
from numpy import pi

from .error import PatternError
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
                 'a_count',
                 'b_count',
                 'identifier')

    pattern: 'Pattern'

    _offset: numpy.ndarray
    _rotation: float
    _dose: float
    _scale: float
    _mirrored: List[bool]

    _a_vector: numpy.ndarray
    _b_vector: numpy.ndarray or None
    a_count: int
    b_count: int

    identifier: Tuple

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
                 scale: float = 1.0):
        """
        :param a_vector: First lattice vector, of the form [x, y].
            Specifies center-to-center spacing between adjacent elements.
        :param a_count: Number of elements in the a_vector direction.
        :param b_vector: Second lattice vector, of the form [x, y].
            Specifies center-to-center spacing between adjacent elements.
            Can be omitted when specifying a 1D array.
        :param b_count: Number of elements in the b_vector direction.
            Should be omitted if b_vector was omitted.
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

    def  __deepcopy__(self, memo: Dict = None) -> 'GridReptition':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new.pattern = copy.deepcopy(self.pattern, memo)
        new._offset = self._offset.copy()
        new._mirrored = copy.deepcopy(self._mirrored, memo)
        new._a_vector = self._a_vector.copy()
        new._b_vector = copy.copy(self._b_vector)       # ndarray or None so don't need deepcopy
        return new

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
    def mirrored(self) -> List[bool]:
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: List[bool]):
        if is_scalar(val):
            raise PatternError('Mirrored must be a 2-element list of booleans')
        self._mirrored = val

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
                newPat = self.pattern.deepcopy()
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
        Rotate around a point

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
        self.rotation += rotation
        return self

    def mirror(self, axis: int) -> 'GridRepetition':
        """
        Mirror the GridRepetition across an axis.

        :param axis: Axis to mirror across.
        :return: self
        """
        self.mirrored[axis] = not self.mirrored[axis]
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

