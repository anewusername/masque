from typing import TypeVar
from abc import ABCMeta, abstractmethod

import numpy
from numpy import pi
from numpy.typing import ArrayLike, NDArray

#from .positionable import Positionable
from ..error import MasqueError
from ..utils import is_scalar, rotation_matrix_2d

T = TypeVar('T', bound='Rotatable')
I = TypeVar('I', bound='RotatableImpl')
P = TypeVar('P', bound='Pivotable')
J = TypeVar('J', bound='PivotableImpl')


class Rotatable(metaclass=ABCMeta):
    """
    Abstract class for all rotatable entities
    """
    __slots__ = ()

    '''
    ---- Abstract methods
    '''
    @abstractmethod
    def rotate(self: T, val: float) -> T:
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
    __slots__ = ()

    _rotation: float
    """ rotation for the object, radians counterclockwise """

    '''
    ---- Properties
    '''
    @property
    def rotation(self) -> float:
        """ Rotation, radians counterclockwise """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float):
        if not numpy.size(val) == 1:
            raise MasqueError('Rotation must be a scalar')
        self._rotation = val % (2 * pi)

    '''
    ---- Methods
    '''
    def rotate(self: I, rotation: float) -> I:
        self.rotation += rotation
        return self

    def set_rotation(self: I, rotation: float) -> I:
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
    Abstract class for entites which can be rotated around a point.
    This requires that they are `Positionable` but not necessarily `Rotatable` themselves.
    """
    __slots__ = ()

    @abstractmethod
    def rotate_around(self: P, pivot: ArrayLike, rotation: float) -> P:
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

    def rotate_around(self: J, pivot: ArrayLike, rotation: float) -> J:
        pivot = numpy.array(pivot, dtype=float)
        self.translate(-pivot)
        self.rotate(rotation)
        self.offset = numpy.dot(rotation_matrix_2d(rotation), self.offset)      #type: ignore #TODO: mypy#3004
        self.translate(+pivot)
        return self

