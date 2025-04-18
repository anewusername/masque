from typing import Any, cast
import copy
import functools

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_VERTICES
from ..error import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, annotations_t, annotations_lt, annotations_eq, rep2key


@functools.total_ordering
class Circle(Shape):
    """
    A circle, which has a position and radius.
    """
    __slots__ = (
        '_radius',
        # Inherited
        '_offset', '_repetition', '_annotations',
        )

    _radius: float
    """ Circle radius """

    # radius property
    @property
    def radius(self) -> float:
        """
        Circle's radius (float, >= 0)
        """
        return self._radius

    @radius.setter
    def radius(self, val: float) -> None:
        if not is_scalar(val):
            raise PatternError('Radius must be a scalar')
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self._radius = val

    def __init__(
            self,
            radius: float,
            *,
            offset: ArrayLike = (0.0, 0.0),
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(offset, numpy.ndarray)
            self._radius = radius
            self._offset = offset
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
        else:
            self.radius = radius
            self.offset = offset
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}

    def __deepcopy__(self, memo: dict | None = None) -> 'Circle':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        new._annotations = copy.deepcopy(self._annotations)
        return new

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) is type(other)
            and numpy.array_equal(self.offset, other.offset)
            and self.radius == other.radius
            and self.repetition == other.repetition
            and annotations_eq(self.annotations, other.annotations)
            )

    def __lt__(self, other: Shape) -> bool:
        if type(self) is not type(other):
            if repr(type(self)) != repr(type(other)):
                return repr(type(self)) < repr(type(other))
            return id(type(self)) < id(type(other))
        other = cast('Circle', other)
        if not self.radius == other.radius:
            return self.radius < other.radius
        if not numpy.array_equal(self.offset, other.offset):
            return tuple(self.offset) < tuple(other.offset)
        if self.repetition != other.repetition:
            return rep2key(self.repetition) < rep2key(other.repetition)
        return annotations_lt(self.annotations, other.annotations)

    def to_polygons(
            self,
            num_vertices: int | None = DEFAULT_POLY_NUM_VERTICES,
            max_arclen: float | None = None,
            ) -> list[Polygon]:
        if (num_vertices is None) and (max_arclen is None):
            raise PatternError('Number of points and arclength left '
                               'unspecified (default was also overridden)')

        n: list[float] = []
        if num_vertices is not None:
            n += [num_vertices]
        if max_arclen is not None:
            n += [2 * pi * self.radius / max_arclen]
        num_vertices = int(round(max(n)))
        thetas = numpy.linspace(2 * pi, 0, num_vertices, endpoint=False)
        xs = numpy.cos(thetas) * self.radius
        ys = numpy.sin(thetas) * self.radius
        xys = numpy.vstack((xs, ys)).T

        return [Polygon(xys, offset=self.offset)]

    def get_bounds_single(self) -> NDArray[numpy.float64]:
        return numpy.vstack((self.offset - self.radius,
                             self.offset + self.radius))

    def rotate(self, theta: float) -> 'Circle':      # noqa: ARG002  (theta unused)
        return self

    def mirror(self, axis: int = 0) -> 'Circle':     # noqa: ARG002  (axis unused)
        self.offset *= -1
        return self

    def scale_by(self, c: float) -> 'Circle':
        self.radius *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        rotation = 0.0
        magnitude = self.radius / norm_value
        return ((type(self),),
                (self.offset, magnitude, rotation, False),
                lambda: Circle(radius=norm_value))

    def __repr__(self) -> str:
        return f'<Circle o{self.offset} r{self.radius:g}>'
