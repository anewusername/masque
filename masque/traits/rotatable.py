from typing import Self, cast, Any
from abc import ABCMeta, abstractmethod

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from .positionable import Positionable
from ..error import MasqueError
from ..utils import rotation_matrix_2d


_empty_slots = ()     # Workaround to get mypy to ignore intentionally empty slots for superclass


class Rotatable(metaclass=ABCMeta):
    """
    Trait class for all rotatable entities
    """
    __slots__ = ()

    #
    # Methods
    #
    @abstractmethod
    def rotate(self, val: float) -> Self:
        """
        Rotate the shape around its origin (0, 0), ignoring its offset.

        Args:
            val: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        pass


class RotatableImpl(Rotatable, metaclass=ABCMeta):
    """
    Simple implementation of `Rotatable`
    """
    __slots__ = _empty_slots

    _rotation: float
    """ rotation for the object, radians counterclockwise """

    #
    # Properties
    #
    @property
    def rotation(self) -> float:
        """ Rotation, radians counterclockwise """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float):
        if not numpy.size(val) == 1:
            raise MasqueError('Rotation must be a scalar')
        self._rotation = val % (2 * pi)

    #
    # Methods
    #
    def rotate(self, rotation: float) -> Self:
        self.rotation += rotation
        return self

    def set_rotation(self, rotation: float) -> Self:
        """
        Set the rotation to a value

        Args:
            rotation: radians ccw

        Returns:
            self
        """
        self.rotation = rotation
        return self


class Pivotable(metaclass=ABCMeta):
    """
    Trait class for entites which can be rotated around a point.
    This requires that they are `Positionable` but not necessarily `Rotatable` themselves.
    """
    __slots__ = ()

    @abstractmethod
    def rotate_around(self, pivot: ArrayLike, rotation: float) -> Self:
        """
        Rotate the object around a point.

        Args:
            pivot: Point (x, y) to rotate around
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        pass


class PivotableImpl(Pivotable, metaclass=ABCMeta):
    """
    Implementation of `Pivotable` for objects which are `Rotatable`
    """
    __slots__ = ()

    offset: Any         # TODO see if we can get around defining `offset` in  PivotableImpl
    """ `[x_offset, y_offset]` """

    def rotate_around(self, pivot: ArrayLike, rotation: float) -> Self:
        pivot = numpy.array(pivot, dtype=float)
        cast(Positionable, self).translate(-pivot)
        cast(Rotatable, self).rotate(rotation)
        self.offset = numpy.dot(rotation_matrix_2d(rotation), self.offset)      # type: ignore # mypy#3004
        cast(Positionable, self).translate(+pivot)
        return self

