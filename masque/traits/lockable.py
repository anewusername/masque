from typing import List, Tuple, Callable, TypeVar, Optional
from abc import ABCMeta, abstractmethod
import copy
import numpy

from ..error import PatternError, PatternLockedError


T = TypeVar('T', bound='Lockable')
I = TypeVar('I', bound='LockableImpl')


class Lockable(metaclass=ABCMeta):
    """
    Abstract class for all lockable entities
    """
    __slots__ = ()

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

    def lock(self: T) -> T:
        """
        Lock the object, disallowing further changes

        Returns:
            self
        """
        pass

    def unlock(self: T) -> T:
        """
        Unlock the object, reallowing changes

        Returns:
            self
        """
        pass


class LockableImpl(Lockable, metaclass=ABCMeta):
    """
    Simple implementation of Lockable
    """
    __slots__ = ()

    locked: bool
    """ If `True`, disallows changes to the object """

    '''
    ---- Non-abstract methods
    '''
    def __setattr__(self, name, value):
        if self.locked and name != 'locked':
            raise PatternLockedError()
        object.__setattr__(self, name, value)

    def lock(self: I) -> I:
        object.__setattr__(self, 'locked', True)
        return self

    def unlock(self: I) -> I:
        object.__setattr__(self, 'locked', False)
        return self
