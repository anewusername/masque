from typing import List, Tuple, Dict
import copy
import numpy
from numpy import pi

from .error import PatternError, PatternLockedError
from .utils import is_scalar, vector2, rotation_matrix_2d


__author__ = 'Jan Petykiewicz'


class Label:
    """
    A text annotation with a position and layer (but no size; it is not drawn)
    """
    __slots__ = ('_offset', '_layer', '_string', 'identifier', 'locked')

    _offset: numpy.ndarray
    """ [x_offset, y_offset] """

    _layer: int or Tuple
    """ Layer (integer >= 0, or 2-Tuple of integers) """

    _string: str
    """ Label string """

    identifier: Tuple
    """ Arbitrary identifier tuple, useful for keeping track of history when flattening """

    locked: bool
    """ If `True`, any changes to the label will raise a `PatternLockedError` """

    def __setattr__(self, name, value):
        if self.locked and name != 'locked':
            raise PatternLockedError()
        object.__setattr__(self, name, value)

    # ---- Properties
    # offset property
    @property
    def offset(self) -> numpy.ndarray:
        """
        [x, y] offset
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
        """
        return self._string

    @string.setter
    def string(self, val: str):
        self._string = val

    def __init__(self,
                 string: str,
                 offset: vector2=(0.0, 0.0),
                 layer: int=0,
                 locked: bool = False):
        self.unlock()
        self.identifier = ()
        self.string = string
        self.offset = numpy.array(offset, dtype=float)
        self.layer = layer
        self.locked = locked

    def __copy__(self) -> 'Label':
        return Label(string=self.string,
                     offset=self.offset.copy(),
                     layer=self.layer,
                     locked=self.locked)

    def  __deepcopy__(self, memo: Dict = None) -> 'Label':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new._offset = self._offset.copy()
        new.locked = self.locked
        return new

    def copy(self) -> 'Label':
        """
        Returns a deep copy of the label.
        """
        return copy.deepcopy(self)

    def translate(self, offset: vector2) -> 'Label':
        """
        Translate the label by the given offset

        Args:
            offset: [x_offset, y,offset]

        Returns:
            self
        """
        self.offset += offset
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'Label':
        """
        Rotate the label around a point.

        Args:
            pivot: Point (x, y) to rotate around
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
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

        Returns:
            Bounds [[xmin, xmax], [ymin, ymax]]
        """
        return numpy.array([self.offset, self.offset])

    def lock(self) -> 'Label':
        """
        Lock the Label, causing any modifications to raise an exception.

        Return:
            self
        """
        object.__setattr__(self, 'locked', True)
        return self

    def unlock(self) -> 'Label':
        """
        Unlock the Label, re-allowing changes.

        Return:
            self
        """
        object.__setattr__(self, 'locked', False)
        return self
