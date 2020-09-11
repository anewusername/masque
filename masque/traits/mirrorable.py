from typing import List, Tuple, Callable, TypeVar, Optional
from abc import ABCMeta, abstractmethod
import copy

from ..error import PatternError, PatternLockedError


T = TypeVar('T', bound='Mirrorable')
#I = TypeVar('I', bound='MirrorableImpl')


class Mirrorable(metaclass=ABCMeta):
    """
    Abstract class for all mirrorable entities
    """
    __slots__ = ()

    '''
    ---- Abstract methods
    '''
    @abstractmethod
    def mirror(self: T, axis: int) -> T:
        """
        Mirror the entity across an axis.

        Args:
            axis: Axis to mirror across.

        Returns:
            self
        """
        pass


#class MirrorableImpl(Mirrorable, metaclass=ABCMeta):
#    """
#    Simple implementation of `Mirrorable`
#    """
#    __slots__ = ()
#
#    _mirrored: numpy.ndarray        # ndarray[bool]
#    """ Whether to mirror the instance across the x and/or y axes. """
#
#    '''
#    ---- Properties
#    '''
#    # Mirrored property
#    @property
#    def mirrored(self) -> numpy.ndarray:        # ndarray[bool]
#        """ Whether to mirror across the [x, y] axes, respectively """
#        return self._mirrored
#
#    @mirrored.setter
#    def mirrored(self, val: Sequence[bool]):
#        if is_scalar(val):
#            raise PatternError('Mirrored must be a 2-element list of booleans')
#        self._mirrored = numpy.array(val, dtype=bool, copy=True)
#
#    '''
#    ---- Methods
#    '''
