from typing import Any, Self
import copy
import math

import numpy
from numpy import pi
from numpy.typing import ArrayLike, NDArray

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_VERTICES
from ..error import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, rotation_matrix_2d, annotations_t


class Ellipse(Shape):
    """
    An ellipse, which has a position, two radii, and a rotation.
    The rotation gives the angle from x-axis, counterclockwise, to the first (x) radius.
    """
    __slots__ = (
        '_radii', '_rotation',
        # Inherited
        '_offset', '_repetition', '_annotations',
        )

    _radii: NDArray[numpy.float64]
    """ Ellipse radii """

    _rotation: float
    """ Angle from x-axis to first radius (ccw, radians) """

    # radius properties
    @property
    def radii(self) -> Any:         # mypy#3004  NDArray[numpy.float64]:
        """
        Return the radii `[rx, ry]`
        """
        return self._radii

    @radii.setter
    def radii(self, val: ArrayLike) -> None:
        val = numpy.array(val).flatten()
        if not val.size == 2:
            raise PatternError('Radii must have length 2')
        if not val.min() >= 0:
            raise PatternError('Radii must be non-negative')
        self._radii = val

    @property
    def radius_x(self) -> float:
        return self.radii[0]

    @radius_x.setter
    def radius_x(self, val: float) -> None:
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self.radii[0] = val

    @property
    def radius_y(self) -> float:
        return self.radii[1]

    @radius_y.setter
    def radius_y(self, val: float) -> None:
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self.radii[1] = val

    # Rotation property
    @property
    def rotation(self) -> float:
        """
        Rotation of rx from the x axis. Uses the interval [0, pi) in radians (counterclockwise
         is positive)

        Returns:
            counterclockwise rotation in radians
        """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float) -> None:
        if not is_scalar(val):
            raise PatternError('Rotation must be a scalar')
        self._rotation = val % pi

    def __init__(
            self,
            radii: ArrayLike,
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0,
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(radii, numpy.ndarray)
            assert isinstance(offset, numpy.ndarray)
            self._radii = radii
            self._offset = offset
            self._rotation = rotation
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
        else:
            self.radii = radii
            self.offset = offset
            self.rotation = rotation
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}

    def __deepcopy__(self, memo: dict | None = None) -> Self:
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        new._radii = self._radii.copy()
        new._annotations = copy.deepcopy(self._annotations)
        return new

    def to_polygons(
            self,
            num_vertices: int | None = DEFAULT_POLY_NUM_VERTICES,
            max_arclen: float | None = None,
            ) -> list[Polygon]:
        if (num_vertices is None) and (max_arclen is None):
            raise PatternError('Number of points and arclength left unspecified'
                               ' (default was also overridden)')

        r0, r1 = self.radii

        # Approximate perimeter
        # Ramanujan, S., "Modular Equations and Approximations to ,"
        #  Quart. J. Pure. Appl. Math., vol. 45 (1913-1914), pp. 350-372
        h = ((r1 - r0) / (r1 + r0)) ** 2
        perimeter = pi * (r1 + r0) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

        n = []
        if num_vertices is not None:
            n += [num_vertices]
        if max_arclen is not None:
            n += [perimeter / max_arclen]
        num_vertices = int(round(max(n)))
        thetas = numpy.linspace(2 * pi, 0, num_vertices, endpoint=False)

        sin_th, cos_th = (numpy.sin(thetas), numpy.cos(thetas))
        xs = r0 * cos_th
        ys = r1 * sin_th
        xys = numpy.vstack((xs, ys)).T

        poly = Polygon(xys, offset=self.offset, rotation=self.rotation)
        return [poly]

    def get_bounds_single(self) -> NDArray[numpy.float64]:
        rot_radii = numpy.dot(rotation_matrix_2d(self.rotation), self.radii)
        return numpy.vstack((self.offset - rot_radii[0],
                             self.offset + rot_radii[1]))

    def rotate(self, theta: float) -> Self:
        self.rotation += theta
        return self

    def mirror(self, axis: int = 0) -> Self:
        self.offset[axis - 1] *= -1
        self.rotation *= -1
        self.rotation += axis * pi
        return self

    def scale_by(self, c: float) -> Self:
        self.radii *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        if self.radius_x < self.radius_y:
            radii = self.radii / self.radius_x
            scale = self.radius_x
            angle = self.rotation
        else:
            radii = self.radii[::-1] / self.radius_y
            scale = self.radius_y
            angle = (self.rotation + pi / 2) % pi
        return ((type(self), radii),
                (self.offset, scale / norm_value, angle, False),
                lambda: Ellipse(radii=radii * norm_value))

    def __repr__(self) -> str:
        rotation = f' r{numpy.rad2deg(self.rotation):g}' if self.rotation != 0 else ''
        return f'<Ellipse o{self.offset} r{self.radii}{rotation}>'
