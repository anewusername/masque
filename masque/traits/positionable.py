# TODO top-level comment about how traits should set __slots__ = (), and how to use AutoSlots

from typing import TypeVar, Any, Optional
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import NDArray, ArrayLike

from ..error import MasqueError


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
    def offset(self) -> NDArray[numpy.float64]:
        """
        [x, y] offset
        """
        pass

#    @offset.setter
#    @abstractmethod
#    def offset(self, val: ArrayLike):
#        pass

    @abstractmethod
    def set_offset(self: T, offset: ArrayLike) -> T:
        """
        Set the offset

        Args:
            offset: [x_offset, y,offset]

        Returns:
            self
        """
        pass

    @abstractmethod
    def translate(self: T, offset: ArrayLike) -> T:
        """
        Translate the entity by the given offset

        Args:
            offset: [x_offset, y,offset]

        Returns:
            self
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Optional[NDArray[numpy.float64]]:
        """
        Returns `[[x_min, y_min], [x_max, y_max]]` which specify a minimal bounding box for the entity.
        Returns `None` for an empty entity.
        """
        pass

    def get_bounds_nonempty(self) -> NDArray[numpy.float64]:
        """
        Returns `[[x_min, y_min], [x_max, y_max]]` which specify a minimal bounding box for the entity.
        Asserts that the entity is non-empty (i.e., `get_bounds()` does not return None).

        This is handy for destructuring like `xy_min, xy_max = entity.get_bounds_nonempty()`
        """
        bounds = self.get_bounds()
        assert(bounds is not None)
        return bounds


class PositionableImpl(Positionable, metaclass=ABCMeta):
    """
    Simple implementation of Positionable
    """
    __slots__ = ()

    _offset: NDArray[numpy.float64]
    """ `[x_offset, y_offset]` """

    '''
    ---- Properties
    '''
    # offset property
    @property
    def offset(self) -> Any:  #TODO mypy#3003  NDArray[numpy.float64]:
        """
        [x, y] offset
        """
        return self._offset

    @offset.setter
    def offset(self, val: ArrayLike) -> None:
        if not isinstance(val, numpy.ndarray) or val.dtype != numpy.float64:
            val = numpy.array(val, dtype=float)

        if val.size != 2:
            raise MasqueError('Offset must be convertible to size-2 ndarray')
        self._offset = val.flatten()

    '''
    ---- Methods
    '''
    def set_offset(self: I, offset: ArrayLike) -> I:
        self.offset = offset
        return self

    def translate(self: I, offset: ArrayLike) -> I:
        self._offset += offset   # type: ignore         # NDArray += ArrayLike should be fine??
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
