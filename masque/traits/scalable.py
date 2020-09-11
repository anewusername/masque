from typing import List, Tuple, Callable, TypeVar, Optional
from abc import ABCMeta, abstractmethod
import copy

from ..error import PatternError, PatternLockedError
from ..utils import is_scalar


T = TypeVar('T', bound='Scalable')
I = TypeVar('I', bound='ScalableImpl')


class Scalable(metaclass=ABCMeta):
    """
    Abstract class for all scalable entities
    """
    __slots__ = ()

    '''
    ---- Abstract methods
    '''
    @abstractmethod
    def scale_by(self: T, c: float) -> T:
        """
        Scale the entity by a factor

        Args:
            c: scaling factor

        Returns:
            self
        """
        pass


class ScalableImpl(Scalable, metaclass=ABCMeta):
    """
    Simple implementation of Scalable
    """
    __slots__ = ()

    _scale: float
    """ scale factor for the entity """

    '''
    ---- Properties
    '''
    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, val: float):
        if not is_scalar(val):
            raise PatternError('Scale must be a scalar')
        if not val > 0:
            raise PatternError('Scale must be positive')
        self._scale = val

    '''
    ---- Methods
    '''
    def scale_by(self: I, c: float) -> I:
        self.scale *= c
        return self

    def set_scale(self: I, scale: float) -> I:
        """
        Set the sclae to a value

        Args:
            scale: absolute scale factor

        Returns:
            self
        """
        self.scale = scale
        return self
