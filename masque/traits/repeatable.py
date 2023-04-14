from typing import Self, TYPE_CHECKING
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import NDArray

from ..error import MasqueError
from .positionable import Bounded


_empty_slots = ()     # Workaround to get mypy to ignore intentionally empty slots for superclass


if TYPE_CHECKING:
    from ..repetition import Repetition


class Repeatable(metaclass=ABCMeta):
    """
    Trait class for all repeatable entities
    """
    __slots__ = ()

    #
    # Properties
    #
    @property
    @abstractmethod
    def repetition(self) -> 'Repetition | None':
        """
        Repetition object, or None (single instance only)
        """
        pass

#    @repetition.setter
#    @abstractmethod
#    def repetition(self, repetition: 'Repetition | None'):
#        pass

    #
    # Methods
    #
    @abstractmethod
    def set_repetition(self, repetition: 'Repetition | None') -> Self:
        """
        Set the repetition

        Args:
            repetition: new value for repetition, or None (single instance)

        Returns:
            self
        """
        pass


class RepeatableImpl(Repeatable, Bounded, metaclass=ABCMeta):
    """
    Simple implementation of `Repeatable` and extension of `Bounded` to include repetition bounds.
    """
    __slots__ = _empty_slots

    _repetition: 'Repetition | None'
    """ Repetition object, or None (single instance only) """

    @abstractmethod
    def get_bounds_single(self, *args, **kwargs) -> NDArray[numpy.float64] | None:
        pass

    #
    # Non-abstract properties
    #
    @property
    def repetition(self) -> 'Repetition | None':
        return self._repetition

    @repetition.setter
    def repetition(self, repetition: 'Repetition | None'):
        from ..repetition import Repetition
        if repetition is not None and not isinstance(repetition, Repetition):
            raise MasqueError(f'{repetition} is not a valid Repetition object!')
        self._repetition = repetition

    #
    # Non-abstract methods
    #
    def set_repetition(self, repetition: 'Repetition | None') -> Self:
        self.repetition = repetition
        return self

    def get_bounds_single_nonempty(self, *args, **kwargs) -> NDArray[numpy.float64]:
        """
        Returns `[[x_min, y_min], [x_max, y_max]]` which specify a minimal bounding box for the entity.
        Asserts that the entity is non-empty (i.e., `get_bounds()` does not return None).

        This is handy for destructuring like `xy_min, xy_max = entity.get_bounds_nonempty()`
        """
        bounds = self.get_bounds_single(*args, **kwargs)
        assert bounds is not None
        return bounds

    def get_bounds(self, *args, **kwargs) -> NDArray[numpy.float64] | None:
        bounds = self.get_bounds_single(*args, **kwargs)

        if bounds is not None and self.repetition is not None:
            rep_bounds = self.repetition.get_bounds()
            if rep_bounds is None:
                return None
            bounds += rep_bounds
        return bounds
