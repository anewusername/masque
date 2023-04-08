from typing import Self
from abc import ABCMeta, abstractmethod

from ..utils import layer_t


_empty_slots = ()     # Workaround to get mypy to ignore intentionally empty slots for superclass


class Layerable(metaclass=ABCMeta):
    """
    Trait class for all layerable entities
    """
    __slots__ = ()

    #
    # Properties
    #
    @property
    @abstractmethod
    def layer(self) -> layer_t:
        """
        Layer number or name (int, tuple of ints, or string)
        """
        pass

#    @layer.setter
#    @abstractmethod
#    def layer(self, val: layer_t):
#        pass

    #
    # Methods
    #
    @abstractmethod
    def set_layer(self, layer: layer_t) -> Self:
        """
        Set the layer

        Args:
            layer: new value for layer

        Returns:
            self
        """
        pass


class LayerableImpl(Layerable, metaclass=ABCMeta):
    """
    Simple implementation of Layerable
    """
    __slots__ = _empty_slots

    _layer: layer_t
    """ Layer number, pair, or name """

    #
    # Non-abstract properties
    #
    @property
    def layer(self) -> layer_t:
        return self._layer

    @layer.setter
    def layer(self, val: layer_t):
        self._layer = val

    #
    # Non-abstract methods
    #
    def set_layer(self, layer: layer_t) -> Self:
        self.layer = layer
        return self
