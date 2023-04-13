"""
    Repetitions provide support for efficiently representing multiple identical
     instances of an object .
"""

from typing import Any, Type
import copy
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import ArrayLike, NDArray

from .traits import Copyable, Scalable, Rotatable, Mirrorable, Bounded
from .error import PatternError
from .utils import rotation_matrix_2d


class Repetition(Copyable, Rotatable, Mirrorable, Scalable, Bounded, metaclass=ABCMeta):
    """
    Interface common to all objects which specify repetitions
    """
    __slots__ = ()      # Allow subclasses to use __slots__

    @property
    @abstractmethod
    def displacements(self) -> NDArray[numpy.float64]:
        """
        An Nx2 ndarray specifying all offsets generated by this repetition
        """
        pass


class Grid(Repetition):
    """
    `Grid` describes a 2D grid formed by two basis vectors and two 'counts' (sizes).

    The second basis vector and count (`b_vector` and `b_count`) may be omitted,
      which makes the grid describe a 1D array.

    Note that the offsets in either the 2D or 1D grids do not have to be axis-aligned.
    """
    __slots__ = (
        '_a_vector', '_b_vector',
        '_a_count', '_b_count',
        )

    _a_vector: NDArray[numpy.float64]
    """ Vector `[x, y]` specifying the first lattice vector of the grid.
        Specifies center-to-center spacing between adjacent elements.
    """

    _a_count: int
    """ Number of instances along the direction specified by the `a_vector` """

    _b_vector: NDArray[numpy.float64] | None
    """ Vector `[x, y]` specifying a second lattice vector for the grid.
        Specifies center-to-center spacing between adjacent elements.
        Can be `None` for a 1D array.
    """

    _b_count: int
    """ Number of instances along the direction specified by the `b_vector` """

    def __init__(
            self,
            a_vector: ArrayLike,
            a_count: int,
            b_vector: ArrayLike | None = None,
            b_count: int | None = 1,
            ) -> None:
        """
        Args:
            a_vector: First lattice vector, of the form `[x, y]`.
                Specifies center-to-center spacing between adjacent instances.
            a_count: Number of elements in the a_vector direction.
            b_vector: Second lattice vector, of the form `[x, y]`.
                Specifies center-to-center spacing between adjacent instances.
                Can be omitted when specifying a 1D array.
            b_count: Number of elements in the `b_vector` direction.
                Should be omitted if `b_vector` was omitted.

        Raises:
            PatternError if `b_*` inputs conflict with each other
            or `a_count < 1`.
        """
        if b_count is None:
            b_count = 1

        if b_vector is None:
            if b_count > 1:
                raise PatternError('Repetition has b_count > 1 but no b_vector')
            else:
                b_vector = numpy.array([0.0, 0.0])

        if a_count < 1:
            raise PatternError(f'Repetition has too-small a_count: {a_count}')
        if b_count < 1:
            raise PatternError(f'Repetition has too-small b_count: {b_count}')

        self.a_vector = a_vector        # type: ignore     # setter handles type conversion
        self.b_vector = b_vector        # type: ignore     # setter handles type conversion
        self.a_count = a_count
        self.b_count = b_count

    @classmethod
    def aligned(
            cls: Type,
            x: float,
            y: float,
            x_count: int,
            y_count: int,
            ) -> 'Grid':
        """
        Simple constructor for an axis-aligned 2D grid

        Args:
            x: X-step
            y: Y-step
            x_count: count of columns
            y_count: count of rows

        Returns:
            An Grid instance with the requested values
        """
        return cls(a_vector=(x, 0), b_vector=(0, y), a_count=x_count, b_count=y_count)

    def __copy__(self) -> 'Grid':
        new = Grid(
            a_vector=self.a_vector.copy(),
            b_vector=copy.copy(self.b_vector),
            a_count=self.a_count,
            b_count=self.b_count,
            )
        return new

    def __deepcopy__(self, memo: dict | None = None) -> 'Grid':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        return new

    # a_vector property
    @property
    def a_vector(self) -> NDArray[numpy.float64]:
        return self._a_vector

    @a_vector.setter
    def a_vector(self, val: ArrayLike) -> None:
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float)

        if val.size != 2:
            raise PatternError('a_vector must be convertible to size-2 ndarray')
        self._a_vector = val.flatten().astype(float)

    # b_vector property
    @property
    def b_vector(self) -> NDArray[numpy.float64] | None:
        return self._b_vector

    @b_vector.setter
    def b_vector(self, val: ArrayLike) -> None:
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float, copy=True)

        if val.size != 2:
            raise PatternError('b_vector must be convertible to size-2 ndarray')
        self._b_vector = val.flatten()

    # a_count property
    @property
    def a_count(self) -> int:
        return self._a_count

    @a_count.setter
    def a_count(self, val: int) -> None:
        if val != int(val):
            raise PatternError('a_count must be convertable to an int!')
        self._a_count = int(val)

    # b_count property
    @property
    def b_count(self) -> int:
        return self._b_count

    @b_count.setter
    def b_count(self, val: int) -> None:
        if val != int(val):
            raise PatternError('b_count must be convertable to an int!')
        self._b_count = int(val)

    @property
    def displacements(self) -> NDArray[numpy.float64]:
        if self.b_vector is None:
            return numpy.arange(self.a_count)[:, None] * self.a_vector[None, :]

        aa, bb = numpy.meshgrid(numpy.arange(self.a_count), numpy.arange(self.b_count), indexing='ij')
        return (aa.flatten()[:, None] * self.a_vector[None, :]
              + bb.flatten()[:, None] * self.b_vector[None, :])             # noqa

    def rotate(self, rotation: float) -> 'Grid':
        """
        Rotate lattice vectors (around (0, 0))

        Args:
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        self.a_vector = numpy.dot(rotation_matrix_2d(rotation), self.a_vector)
        if self.b_vector is not None:
            self.b_vector = numpy.dot(rotation_matrix_2d(rotation), self.b_vector)
        return self

    def mirror(self, axis: int) -> 'Grid':
        """
        Mirror the Grid across an axis.

        Args:
            axis: Axis to mirror across.
                (0: mirror across x-axis, 1: mirror across y-axis)

        Returns:
            self
        """
        self.a_vector[1 - axis] *= -1
        if self.b_vector is not None:
            self.b_vector[1 - axis] *= -1
        return self

    def get_bounds(self) -> NDArray[numpy.float64] | None:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the `Grid` in each dimension.

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        a_extent = self.a_vector * self.a_count
        b_extent = self.b_vector * self.b_count if (self.b_vector is not None) else 0       # type: NDArray[numpy.float64] | float

        corners = numpy.stack(((0, 0), a_extent, b_extent, a_extent + b_extent))
        xy_min = numpy.min(corners, axis=0)
        xy_max = numpy.max(corners, axis=0)
        return numpy.array((xy_min, xy_max))

    def scale_by(self, c: float) -> 'Grid':
        """
        Scale the Grid by a factor

        Args:
            c: scaling factor

        Returns:
            self
        """
        self.a_vector *= c
        if self.b_vector is not None:
            self.b_vector *= c
        return self

    def __repr__(self) -> str:
        bv = f', {self.b_vector}' if self.b_vector is not None else ''
        return (f'<Grid {self.a_count}x{self.b_count} ({self.a_vector}{bv})>')

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.a_count != other.a_count or self.b_count != other.b_count:
            return False
        if any(self.a_vector[ii] != other.a_vector[ii] for ii in range(2)):
            return False
        if self.b_vector is None and other.b_vector is None:
            return True
        if self.b_vector is None or other.b_vector is None:
            return False
        if any(self.b_vector[ii] != other.b_vector[ii] for ii in range(2)):
            return False
        return True


class Arbitrary(Repetition):
    """
    `Arbitrary` is a simple list of (absolute) displacements for instances.

    Attributes:
        displacements (numpy.ndarray): absolute displacements of all elements
                                       `[[x0, y0], [x1, y1], ...]`
    """

    __slots__ = ('_displacements',)

    _displacements: NDArray[numpy.float64]
    """ List of vectors `[[x0, y0], [x1, y1], ...]` specifying the offsets
          of the instances.
    """

    @property
    def displacements(self) -> Any:     # TODO: mypy#3004   NDArray[numpy.float64]:
        return self._displacements

    @displacements.setter
    def displacements(self, val: ArrayLike) -> None:
        vala: NDArray[numpy.float64] = numpy.array(val, dtype=float)
        vala = numpy.sort(vala.view([('', vala.dtype)] * vala.shape[1]), 0).view(vala.dtype)    # sort rows
        self._displacements = vala

    def __init__(
            self,
            displacements: ArrayLike,
            ) -> None:
        """
        Args:
            displacements: List of vectors (Nx2 ndarray) specifying displacements.
        """
        self.displacements = displacements

    def __repr__(self) -> str:
        return (f'<Arbitrary {len(self.displacements)}pts >')

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return numpy.array_equal(self.displacements, other.displacements)

    def rotate(self, rotation: float) -> 'Arbitrary':
        """
        Rotate dispacements (around (0, 0))

        Args:
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        self.displacements = numpy.dot(rotation_matrix_2d(rotation), self.displacements.T).T
        return self

    def mirror(self, axis: int) -> 'Arbitrary':
        """
        Mirror the displacements across an axis.

        Args:
            axis: Axis to mirror across.
                (0: mirror across x-axis, 1: mirror across y-axis)

        Returns:
            self
        """
        self.displacements[1 - axis] *= -1
        return self

    def get_bounds(self) -> NDArray[numpy.float64] | None:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the `displacements` in each dimension.

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        xy_min = numpy.min(self.displacements, axis=0)
        xy_max = numpy.max(self.displacements, axis=0)
        return numpy.array((xy_min, xy_max))

    def scale_by(self, c: float) -> 'Arbitrary':
        """
        Scale the displacements by a factor

        Args:
            c: scaling factor

        Returns:
            self
        """
        self.displacements *= c
        return self

