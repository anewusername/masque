from typing import Self, TYPE_CHECKING
from abc import ABCMeta, abstractmethod

from ..error import MasqueError


_empty_slots = ()     # Workaround to get mypy to ignore intentionally empty slots for superclass


if TYPE_CHECKING:
    from ..repetition import Repetition


class Repeatable(metaclass=ABCMeta):
    """
    Abstract class for all repeatable entities
    """
    __slots__ = ()

    '''
    ---- Properties
    '''
    @property
    @abstractmethod
    def repetition(self) -> 'Repetition' | None:
        """
        Repetition object, or None (single instance only)
        """
        pass

#    @repetition.setter
#    @abstractmethod
#    def repetition(self, repetition: 'Repetition' | None):
#        pass

    '''
    ---- Methods
    '''
    @abstractmethod
    def set_repetition(self, repetition: 'Repetition' | None) -> Self:
        """
        Set the repetition

        Args:
            repetition: new value for repetition, or None (single instance)

        Returns:
            self
        """
        pass


class RepeatableImpl(Repeatable, metaclass=ABCMeta):
    """
    Simple implementation of `Repeatable`
    """
    __slots__ = _empty_slots

    _repetition: 'Repetition' | None
    """ Repetition object, or None (single instance only) """

    '''
    ---- Non-abstract properties
    '''
    @property
    def repetition(self) -> 'Repetition' | None:
        return self._repetition

    @repetition.setter
    def repetition(self, repetition: 'Repetition' | None):
        from ..repetition import Repetition
        if repetition is not None and not isinstance(repetition, Repetition):
            raise MasqueError(f'{repetition} is not a valid Repetition object!')
        self._repetition = repetition

    '''
    ---- Non-abstract methods
    '''
    def set_repetition(self, repetition: 'Repetition' | None) -> Self:
        self.repetition = repetition
        return self
