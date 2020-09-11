from typing import List, Tuple, Callable, TypeVar, Optional
from abc import ABCMeta, abstractmethod
import copy

from ..error import PatternError, PatternLockedError
from ..utils import is_scalar


T = TypeVar('T', bound='Doseable')
I = TypeVar('I', bound='DoseableImpl')


class Doseable(metaclass=ABCMeta):
    """
    Abstract class for all doseable entities
    """
    __slots__ = ()

    '''
    ---- Properties
    '''
    @property
    @abstractmethod
    def dose(self) -> float:
        """
        Dose (float >= 0)
        """
        pass

#    @dose.setter
#    @abstractmethod
#    def dose(self, val: float):
#        pass

    '''
    ---- Methods
    '''
    def set_dose(self: T, dose: float) -> T:
        """
        Set the dose

        Args:
            dose: new value for dose

        Returns:
            self
        """
        pass


class DoseableImpl(Doseable, metaclass=ABCMeta):
    """
    Simple implementation of Doseable
    """
    __slots__ = ()

    _dose: float
    """ Dose """

    '''
    ---- Non-abstract properties
    '''
    @property
    def dose(self) -> float:
        return self._dose

    @dose.setter
    def dose(self, val: float):
        if not val >= 0:
            raise PatternError('Dose must be non-negative')
        self._dose = val


    '''
    ---- Non-abstract methods
    '''
    def set_dose(self: I, dose: float) -> I:
        self.dose = dose
        return self
