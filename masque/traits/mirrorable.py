from typing import Self
from abc import ABCMeta, abstractmethod


class Mirrorable(metaclass=ABCMeta):
    """
    Trait class for all mirrorable entities
    """
    __slots__ = ()

    @abstractmethod
    def mirror(self, axis: int) -> Self:
        """
        Mirror the entity across an axis.

        Args:
            axis: Axis to mirror across.

        Returns:
            self
        """
        pass

    def mirror2d(self, axes: tuple[bool, bool]) -> Self:
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
#    #
#    # Properties
#    #
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
#    #
#    # Methods
#    #
