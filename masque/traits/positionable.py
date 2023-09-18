from typing import Self, Any
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import NDArray, ArrayLike

from ..error import MasqueError


_empty_slots = ()     # Workaround to get mypy to ignore intentionally empty slots for superclass


class Positionable(metaclass=ABCMeta):
    """
    Trait class for all positionable entities
    """
    __slots__ = ()

    #
    # Properties
    #
    @property
    @abstractmethod
    def offset(self) -> NDArray[numpy.float64]:
        """
        [x, y] offset
        """
        pass

    @offset.setter
    @abstractmethod
    def offset(self, val: ArrayLike) -> None:
        pass

    @abstractmethod
    def set_offset(self, offset: ArrayLike) -> Self:
        """
        Set the offset

        Args:
            offset: [x_offset, y,offset]

        Returns:
            self
        """
        pass

    @abstractmethod
    def translate(self, offset: ArrayLike) -> Self:
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
    __slots__ = _empty_slots

    _offset: NDArray[numpy.float64]
    """ `[x_offset, y_offset]` """

    #
    # Properties
    #
    # offset property
    @property
    def offset(self) -> Any:  # mypy#3004  NDArray[numpy.float64]:
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
        self._offset = val.flatten()        # type: ignore

    #
    # Methods
    #
    def set_offset(self, offset: ArrayLike) -> Self:
        self.offset = offset
        return self

    def translate(self, offset: ArrayLike) -> Self:
        self._offset += offset   # type: ignore         # NDArray += ArrayLike should be fine??
        return self


class Bounded(metaclass=ABCMeta):
    @abstractmethod
    def get_bounds(self, *args, **kwargs) -> NDArray[numpy.float64] | None:
        """
        Returns `[[x_min, y_min], [x_max, y_max]]` which specify a minimal bounding box for the entity.
        Returns `None` for an empty entity.
        """
        pass

    def get_bounds_nonempty(self, *args, **kwargs) -> NDArray[numpy.float64]:
        """
        Returns `[[x_min, y_min], [x_max, y_max]]` which specify a minimal bounding box for the entity.
        Asserts that the entity is non-empty (i.e., `get_bounds()` does not return None).

        This is handy for destructuring like `xy_min, xy_max = entity.get_bounds_nonempty()`
        """
        bounds = self.get_bounds(*args, **kwargs)
        assert bounds is not None
        return bounds


