from typing import List, Tuple, Callable, TypeVar, Optional
from abc import ABCMeta, abstractmethod
import copy


T = TypeVar('T', bound='Copyable')


class Copyable(metaclass=ABCMeta):
    """
    Abstract class which adds .copy() and .deepcopy()
    """
    __slots__ = ()

    '''
    ---- Non-abstract methods
    '''
    def copy(self: T) -> T:
        """
        Return a shallow copy of the object.

        Returns:
            `copy.copy(self)`
        """
        return copy.copy(self)

    def deepcopy(self: T) -> T:
        """
        Return a deep copy of the object.

        Returns:
            `copy.deepcopy(self)`
        """
        return copy.deepcopy(self)
