from typing import List, Tuple, Callable, TypeVar, Optional, TYPE_CHECKING
from abc import ABCMeta, abstractmethod
import copy

from ..error import PatternError, PatternLockedError


if TYPE_CHECKING:
    from ..repetition import Repetition


T = TypeVar('T', bound='Repeatable')
I = TypeVar('I', bound='RepeatableImpl')


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
    def repetition(self) -> Optional['Repetition']:
        """
        Repetition object, or None (single instance only)
        """
        pass

#    @repetition.setter
#    @abstractmethod
#    def repetition(self, repetition: Optional['Repetition']):
#        pass

    '''
    ---- Methods
    '''
    @abstractmethod
    def set_repetition(self: T, repetition: Optional['Repetition']) -> T:
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
    __slots__ = ()

    _repetition: Optional['Repetition']
    """ Repetition object, or None (single instance only) """

    '''
    ---- Non-abstract properties
    '''
    @property
    def repetition(self) -> Optional['Repetition']:
        return self._repetition

    @repetition.setter
    def repetition(self, repetition: Optional['Repetition']):
        from ..repetition import Repetition
        if repetition is not None and not isinstance(repetition, Repetition):
            raise PatternError(f'{repetition} is not a valid Repetition object!')
        self._repetition = repetition

    '''
    ---- Non-abstract methods
    '''
    def set_repetition(self: I, repetition: Optional['Repetition']) -> I:
        self.repetition = repetition
        return self
