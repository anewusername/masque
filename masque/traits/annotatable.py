#from types import MappingProxyType
from abc import ABCMeta, abstractmethod

from ..utils import annotations_t
from ..error import MasqueError


_empty_slots = ()     # Workaround to get mypy to ignore intentionally empty slots for superclass


class Annotatable(metaclass=ABCMeta):
    """
    Trait class for all annotatable entities
    Annotations correspond to GDS/OASIS "properties"
    """
    __slots__ = ()

    #
    # Properties
    #
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
    __slots__ = _empty_slots

    _annotations: annotations_t
    """ Dictionary storing annotation name/value pairs """

    #
    # Non-abstract properties
    #
    @property
    def annotations(self) -> annotations_t:
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: annotations_t) -> None:
        if not isinstance(annotations, dict):
            raise MasqueError(f'annotations expected dict, got {type(annotations)}')
        self._annotations = annotations
