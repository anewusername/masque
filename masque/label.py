from typing import List, Tuple, Dict
import copy
import numpy
from numpy import pi

from .error import PatternError, PatternLockedError
from .utils import is_scalar, vector2, rotation_matrix_2d, layer_t, AutoSlots
from .traits import PositionableImpl, LayerableImpl, Copyable, Pivotable, LockableImpl


class Label(PositionableImpl, LayerableImpl, LockableImpl, Pivotable, Copyable, metaclass=AutoSlots):
    """
    A text annotation with a position and layer (but no size; it is not drawn)
    """
    __slots__ = ( '_string', 'identifier')

    _string: str
    """ Label string """

    identifier: Tuple
    """ Arbitrary identifier tuple, useful for keeping track of history when flattening """

    '''
    ---- Properties
    '''
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
                 offset: vector2 = (0.0, 0.0),
                 layer: layer_t = 0,
                 locked: bool = False):
        object.__setattr__(self, 'locked', False)
        self.identifier = ()
        self.string = string
        self.offset = numpy.array(offset, dtype=float, copy=True)
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
        PositionableImpl._lock(self)
        LockableImpl.lock(self)
        return self

    def unlock(self) -> 'Label':
        LockableImpl.unlock(self)
        PositionableImpl._unlock(self)
        return self

    def __repr__(self) -> str:
        locked = ' L' if self.locked else ''
        return f'<Label "{self.string}" l{self.layer} o{self.offset}{locked}>'
