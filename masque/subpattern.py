"""
 SubPattern provides basic support for nesting Pattern objects within each other, by adding
  offset, rotation, scaling, and other such properties to the reference.
"""

from typing import Union, List
import copy

import numpy
from numpy import pi

from .error import PatternError
from .utils import is_scalar, rotation_matrix_2d, vector2


__author__ = 'Jan Petykiewicz'


class SubPattern:
    """
    SubPattern provides basic support for nesting Pattern objects within each other, by adding
     offset, rotation, scaling, and associated methods.
    """

    pattern = None          # type: Pattern
    _offset = (0.0, 0.0)    # type: numpy.ndarray
    _rotation = 0.0         # type: float
    _dose = 1.0             # type: float
    _scale = 1.0            # type: float
    _mirrored = None        # type: List[bool]

    def __init__(self,
                 pattern: 'Pattern',
                 offset: vector2=(0.0, 0.0),
                 rotation: float=0.0,
                 mirrored: List[bool]=None,
                 dose: float=1.0,
                 scale: float=1.0):
        self.pattern = pattern
        self.offset = offset
        self.rotation = rotation
        self.dose = dose
        self.scale = scale
        if mirrored is None:
            mirrored = [False, False]
        self.mirrored = mirrored

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
        self._offset = val.flatten()

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

    def as_pattern(self) -> 'Pattern':
        """
        Returns a copy of self.pattern which has been scaled, rotated, etc. according to this
         SubPattern's properties.
        :return: Copy of self.pattern that has been altered to reflect the SubPattern's properties.
        """
        pattern = self.pattern.deepcopy()
        pattern.scale_by(self.scale)
        [pattern.mirror(ax) for ax, do in enumerate(self.mirrored) if do]
        pattern.rotate_around((0.0, 0.0), self.rotation)
        pattern.translate_elements(self.offset)
        pattern.scale_element_doses(self.dose)
        return pattern

    def translate(self, offset: vector2) -> 'SubPattern':
        """
        Translate by the given offset

        :param offset: Translate by this offset
        :return: self
        """
        self.offset += offset
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'SubPattern':
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

    def rotate(self, rotation: float) -> 'SubPattern':
        """
        Rotate around (0, 0)

        :param rotation: Angle to rotate by (counterclockwise, radians)
        :return: self
        """
        self.rotation += rotation
        return self

    def mirror(self, axis: int) -> 'SubPattern':
        """
        Mirror the subpattern across an axis.

        :param axis: Axis to mirror across.
        :return: self
        """
        self.mirrored[axis] = not self.mirrored[axis]
        return self

    def get_bounds(self) -> numpy.ndarray or None:
        """
        Return a numpy.ndarray containing [[x_min, y_min], [x_max, y_max]], corresponding to the
         extent of the SubPattern in each dimension.
        Returns None if the contained Pattern is empty.

        :return: [[x_min, y_min], [x_max, y_max]] or None
        """
        return self.as_pattern().get_bounds()

    def scale_by(self, c: float) -> 'SubPattern':
        """
        Scale the subpattern by a factor

        :param c: scaling factor
        """
        self.scale *= c
        return self

    def copy(self) -> 'SubPattern':
        """
        Return a shallow copy of the subpattern.

        :return: copy.copy(self)
        """
        return copy.copy(self)

    def deepcopy(self) -> 'SubPattern':
        """
        Return a deep copy of the subpattern.

        :return: copy.copy(self)
        """
        return copy.deepcopy(self)
