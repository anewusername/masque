from typing import List, Tuple, Callable, TypeVar, Optional
from abc import ABCMeta, abstractmethod
import copy

from ..error import PatternError, PatternLockedError
from ..utils import layer_t


T = TypeVar('T', bound='Layerable')
I = TypeVar('I', bound='LayerableImpl')


class Layerable(metaclass=ABCMeta):
    """
    Abstract class for all layerable entities
    """
    __slots__ = ()
    '''
    ---- Properties
    '''
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

    '''
    ---- Methods
    '''
    def set_layer(self: T, layer: layer_t) -> T:
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
    __slots__ = ()

    _layer: layer_t
    """ Layer number, pair, or name """

    '''
    ---- Non-abstract properties
    '''
    @property
    def layer(self) -> layer_t:
        return self._layer

    @layer.setter
    def layer(self, val: layer_t):
        self._layer = val

    '''
    ---- Non-abstract methods
    '''
    def set_layer(self: I, layer: layer_t) -> I:
        self.layer = layer
        return self
