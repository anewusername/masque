from typing import TypeVar, Tuple
from abc import ABCMeta, abstractmethod


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

    def mirror2d(self: T, axes: Tuple[bool, bool]) -> T:
        """
        Optionally mirror the entity across both axes

        Args:
            axes: (mirror_across_x, mirror_across_y)

        Returns:
            self
        """
        if axes[0]:
            self.mirror(0)
        if axes[1]:
            self.mirror(1)
        return self


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
#            raise MasqueError('Mirrored must be a 2-element list of booleans')
#        self._mirrored = numpy.array(val, dtype=bool, copy=True)
#
#    '''
#    ---- Methods
#    '''
