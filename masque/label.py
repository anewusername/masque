from typing import List, Tuple
import copy
import numpy
from numpy import pi

from . import PatternError
from .utils import is_scalar, vector2, rotation_matrix_2d


__author__ = 'Jan Petykiewicz'


class Label:
    """
    A circle, which has a position and radius.
    """

    # [x_offset, y_offset]
    _offset = numpy.array([0.0, 0.0])   # type: numpy.ndarray

    # Layer (integer >= 0)
    _layer = 0                          # type: int or Tuple

    # Label string
    _string = None                      # type: str

    # ---- Properties
    # offset property
    @property
    def offset(self) -> numpy.ndarray:
        """
        [x, y] offset

        :return: [x_offset, y_offset]
        """
        return self._offset

    @offset.setter
    def offset(self, val: vector2):
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float)

        if val.size != 2:
            raise PatternError('Offset must be convertible to size-2 ndarray')
        self._offset = val.flatten()

    # layer property
    @property
    def layer(self) -> int or Tuple[int]:
        """
        Layer number (int or tuple of ints)

        :return: Layer
        """
        return self._layer

    @layer.setter
    def layer(self, val: int or List[int]):
        self._layer = val

    # string property
    @property
    def string(self) -> str:
        """
        Label string (str)

        :return: string
        """
        return self._string

    @string.setter
    def string(self, val: str):
        self._string = val

    def __init__(self,
                 string: str,
                 offset: vector2=(0.0, 0.0),
                 layer: int=0):
        self.string = string
        self.offset = numpy.array(offset, dtype=float)
        self.layer = layer


    # ---- Non-abstract methods
    def copy(self) -> 'Label':
        """
        Returns a deep copy of the shape.

        :return: Deep copy of self
        """
        return copy.deepcopy(self)

    def translate(self, offset: vector2) -> 'Label':
        """
        Translate the shape by the given offset

        :param offset: [x_offset, y,offset]
        :return: self
        """
        self.offset += offset
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'Label':
        """
        Rotate the shape around a point.

        :param pivot: Point (x, y) to rotate around
        :param rotation: Angle to rotate by (counterclockwise, radians)
        :return: self
        """
        pivot = numpy.array(pivot, dtype=float)
        self.translate(-pivot)
        self.offset = numpy.dot(rotation_matrix_2d(rotation), self.offset)
        self.translate(+pivot)
        return self

    def get_bounds(self) -> numpy.ndarray:
        """
        Return the bounds of the label.

        Labels are assumed to take up 0 area, i.e.
        bounds = [self.offset,
                  self.offset]

        :return: Bounds [[xmin, xmax], [ymin, ymax]]
        """
        return numpy.array([self.offset, self.offset])


