from typing import List
import numpy
from numpy import pi

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_POINTS
from .. import PatternError
from ..utils import is_scalar, vector2


__author__ = 'Jan Petykiewicz'


class Circle(Shape):
    """
    A circle, which has a position and radius.
    """

    _radius = None                              # type: float

    # Defaults for to_polygons
    poly_num_points = DEFAULT_POLY_NUM_POINTS   # type: int
    poly_max_arclen = None                      # type: float

    # radius property
    @property
    def radius(self) -> float:
        """
        Circle's radius (float, >= 0)

        :return: radius
        """
        return self._radius

    @radius.setter
    def radius(self, val: float):
        if not is_scalar(val):
            raise PatternError('Radius must be a scalar')
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self._radius = val

    def __init__(self,
                 radius: float,
                 poly_num_points: int=DEFAULT_POLY_NUM_POINTS,
                 poly_max_arclen: float=None,
                 offset: vector2=(0.0, 0.0),
                 layer: int=0,
                 dose: float=1.0):
        self.offset = numpy.array(offset, dtype=float)
        self.layer = layer
        self.dose = dose
        self.radius = radius
        self.poly_num_points = poly_num_points
        self.poly_max_arclen = poly_max_arclen

    def to_polygons(self, poly_num_points: int=None, poly_max_arclen: float=None) -> List[Polygon]:
        if poly_num_points is None:
            poly_num_points = self.poly_num_points
        if poly_max_arclen is None:
            poly_max_arclen = self.poly_max_arclen

        if (poly_num_points is None) and (poly_max_arclen is None):
            raise PatternError('Number of points and arclength left '
                               'unspecified (default was also overridden)')

        n = []
        if poly_num_points is not None:
            n += [poly_num_points]
        if poly_max_arclen is not None:
            n += [2 * pi * self.radius / poly_max_arclen]
        thetas = numpy.linspace(2 * pi, 0, max(n), endpoint=False)
        xs = numpy.cos(thetas) * self.radius
        ys = numpy.sin(thetas) * self.radius
        xys = numpy.vstack((xs, ys)).T

        return [Polygon(xys, offset=self.offset, dose=self.dose, layer=self.layer)]

    def get_bounds(self) -> numpy.ndarray:
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
        return (type(self), self.layer), \
               (self.offset, magnitude, rotation, self.dose), \
               lambda: Circle(radius=norm_value, layer=self.layer)

