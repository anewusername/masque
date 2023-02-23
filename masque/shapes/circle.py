from typing import List, Dict, Optional
import copy

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_VERTICES
from ..error import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, layer_t, annotations_t


class Circle(Shape):
    """
    A circle, which has a position and radius.
    """
    __slots__ = (
        '_radius',
        # Inherited
        '_offset', '_layer', '_repetition', '_annotations',
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
            layer: layer_t = 0,
            repetition: Optional[Repetition] = None,
            annotations: Optional[annotations_t] = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(offset, numpy.ndarray)
            self._radius = radius
            self._offset = offset
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
            self._layer = layer
        else:
            self.radius = radius
            self.offset = offset
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
            self.layer = layer

    def __deepcopy__(self, memo: Optional[Dict] = None) -> 'Circle':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        new._annotations = copy.deepcopy(self._annotations)
        return new

    def to_polygons(
            self,
            num_vertices: Optional[int] = DEFAULT_POLY_NUM_VERTICES,
            max_arclen: Optional[float] = None,
            ) -> List[Polygon]:
        if (num_vertices is None) and (max_arclen is None):
            raise PatternError('Number of points and arclength left '
                               'unspecified (default was also overridden)')

        n: List[float] = []
        if num_vertices is not None:
            n += [num_vertices]
        if max_arclen is not None:
            n += [2 * pi * self.radius / max_arclen]
        num_vertices = int(round(max(n)))
        thetas = numpy.linspace(2 * pi, 0, num_vertices, endpoint=False)
        xs = numpy.cos(thetas) * self.radius
        ys = numpy.sin(thetas) * self.radius
        xys = numpy.vstack((xs, ys)).T

        return [Polygon(xys, offset=self.offset, layer=self.layer)]

    def get_bounds(self) -> NDArray[numpy.float64]:
        return numpy.vstack((self.offset - self.radius,
                             self.offset + self.radius))

    def rotate(self, theta: float) -> 'Circle':
        return self

    def mirror(self, axis: int) -> 'Circle':
        self.offset *= -1
        return self

    def scale_by(self, c: float) -> 'Circle':
        self.radius *= c
        return self

    def normalized_form(self, norm_value) -> normalized_shape_tuple:
        rotation = 0.0
        magnitude = self.radius / norm_value
        return ((type(self), self.layer),
                (self.offset, magnitude, rotation, False),
                lambda: Circle(radius=norm_value, layer=self.layer))

    def __repr__(self) -> str:
        return f'<Circle l{self.layer} o{self.offset} r{self.radius:g}>'
