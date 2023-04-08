from typing import Self
from abc import ABCMeta, abstractmethod

from ..error import MasqueError
from ..utils import is_scalar


_empty_slots = ()     # Workaround to get mypy to ignore intentionally empty slots for superclass


class Scalable(metaclass=ABCMeta):
    """
    Trait class for all scalable entities
    """
    __slots__ = ()

    #
    # Methods
    #
    @abstractmethod
    def scale_by(self, c: float) -> Self:
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
    __slots__ = _empty_slots

    _scale: float
    """ scale factor for the entity """

    #
    # Properties
    #
    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, val: float):
        if not is_scalar(val):
            raise MasqueError('Scale must be a scalar')
        if not val > 0:
            raise MasqueError('Scale must be positive')
        self._scale = val

    #
    # Methods
    #
    def scale_by(self, c: float) -> Self:
        self.scale *= c
        return self

    def set_scale(self, scale: float) -> Self:
        """
        Set the sclae to a value

        Args:
            scale: absolute scale factor

        Returns:
            self
        """
        self.scale = scale
        return self
