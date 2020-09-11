# TODO top-level comment about how traits should set __slots__ = (), and how to use AutoSlots

from typing import List, Tuple, Callable, TypeVar, Optional
from abc import ABCMeta, abstractmethod
import copy
import numpy        # type: ignore

from ..error import PatternError, PatternLockedError
from ..utils import is_scalar, rotation_matrix_2d, vector2


T = TypeVar('T', bound='Positionable')
I = TypeVar('I', bound='PositionableImpl')


class Positionable(metaclass=ABCMeta):
    """
    Abstract class for all positionable entities
    """
    __slots__ = ()

    '''
    ---- Abstract properties
    '''
    @property
    @abstractmethod
    def offset(self) -> numpy.ndarray:
        """
        [x, y] offset
        """
        pass

#    @offset.setter
#    @abstractmethod
#    def offset(self, val: vector2):
#        pass

    '''
    --- Abstract methods
    '''
    @abstractmethod
    def get_bounds(self) -> numpy.ndarray:
        """
        Returns `[[x_min, y_min], [x_max, y_max]]` which specify a minimal bounding box for the entity.
        """
        pass

    @abstractmethod
    def set_offset(self: T, offset: vector2) -> T:
        """
        Set the offset

        Args:
            offset: [x_offset, y,offset]

        Returns:
            self
        """
        pass

    @abstractmethod
    def translate(self: T, offset: vector2) -> T:
        """
        Translate the entity by the given offset

        Args:
            offset: [x_offset, y,offset]

        Returns:
            self
        """
        pass


class PositionableImpl(Positionable, metaclass=ABCMeta):
    """
    Simple implementation of Positionable
    """
    __slots__ = ()

    _offset: numpy.ndarray
    """ `[x_offset, y_offset]` """

    '''
    ---- Properties
    '''
    # offset property
    @property
    def offset(self) -> numpy.ndarray:
        """
        [x, y] offset
        """
        return self._offset

    @offset.setter
    def offset(self, val: vector2):
        if not isinstance(val, numpy.ndarray) or val.dtype != numpy.float64:
            val = numpy.array(val, dtype=float)

        if val.size != 2:
            raise PatternError('Offset must be convertible to size-2 ndarray')
        self._offset = val.flatten()


    '''
    ---- Methods
    '''
    def set_offset(self: I, offset: vector2) -> I:
        self.offset = offset
        return self

    def translate(self: I, offset: vector2) -> I:
        self._offset += offset
        return self

    def _lock(self: I) -> I:
        """
        Lock the entity, disallowing further changes

        Returns:
            self
        """
        self._offset.flags.writeable = False
        return self

    def _unlock(self: I) -> I:
        """
        Unlock the entity

        Returns:
            self
        """
        self._offset.flags.writeable = True
        return self
