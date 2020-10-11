from typing import TypeVar
from types import MappingProxyType
from abc import ABCMeta, abstractmethod
import copy

from ..utils import annotations_t
from ..error import PatternError


T = TypeVar('T', bound='Annotatable')
I = TypeVar('I', bound='AnnotatableImpl')


class Annotatable(metaclass=ABCMeta):
    """
    Abstract class for all annotatable entities
    Annotations correspond to GDS/OASIS "properties"
    """
    __slots__ = ()

    '''
    ---- Properties
    '''
    @property
    @abstractmethod
    def annotations(self) -> annotations_t:
        """
        Dictionary mapping annotation names to values
        """
        pass


class AnnotatableImpl(Annotatable, metaclass=ABCMeta):
    """
    Simple implementation of `Annotatable`.
    """
    __slots__ = ()

    _annotations: annotations_t
    """ Dictionary storing annotation name/value pairs """

    '''
    ---- Non-abstract properties
    '''
    @property
    def annotations(self) -> annotations_t:
#        # TODO: Find a way to make sure the subclass implements Lockable without dealing with diamond inheritance or this extra hasattr
#        if hasattr(self, 'is_locked') and self.is_locked():
#            return MappingProxyType(self._annotations)
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: annotations_t):
        if not isinstance(annotations, dict):
            raise PatternError(f'annotations expected dict, got {type(annotations)}')
        self._annotations = annotations
