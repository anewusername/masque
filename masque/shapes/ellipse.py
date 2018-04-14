from typing import List
import math
import numpy
from numpy import pi

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_POINTS
from .. import PatternError
from ..utils import is_scalar, rotation_matrix_2d, vector2


__author__ = 'Jan Petykiewicz'


class Ellipse(Shape):
    """
    An ellipse, which has a position, two radii, and a rotation.
    The rotation gives the angle from x-axis, counterclockwise, to the first (x) radius.
    """

    _radii = None       # type: numpy.ndarray
    _rotation = 0.0     # type: float

    # Defaults for to_polygons
    poly_num_points = DEFAULT_POLY_NUM_POINTS   # type: int
    poly_max_arclen = None                      # type: float

    # radius properties
    @property
    def radii(self) -> numpy.ndarray:
        """
        Return the radii [rx, ry]

        :return: [rx, ry]
        """
        return self._radii

    @radii.setter
    def radii(self, val: vector2):
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
    def radius_x(self, val: float):
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self.radii[0] = val

    @property
    def radius_y(self) -> float:
        return self.radii[1]

    @radius_y.setter
    def radius_y(self, val: float):
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self.radii[1] = val

    # Rotation property
    @property
    def rotation(self) -> float:
        """
        Rotation of rx from the x axis. Uses the interval [0, pi) in radians (counterclockwise
         is positive)

        :return: counterclockwise rotation in radians
        """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float):
        if not is_scalar(val):
            raise PatternError('Rotation must be a scalar')
        self._rotation = val % pi

    def __init__(self,
                 radii: vector2,
                 rotation: float=0,
                 poly_num_points: int=DEFAULT_POLY_NUM_POINTS,
                 poly_max_arclen: float=None,
                 offset: vector2=(0.0, 0.0),
                 layer: int=0,
                 dose: float=1.0):
        self.offset = offset
        self.layer = layer
        self.dose = dose
        self.radii = radii
        self.rotation = rotation
        self.poly_num_points = poly_num_points
        self.poly_max_arclen = poly_max_arclen

    def to_polygons(self,
                    poly_num_points: int=None,
                    poly_max_arclen: float=None
                    ) -> List[Polygon]:
        if poly_num_points is None:
            poly_num_points = self.poly_num_points
        if poly_max_arclen is None:
            poly_max_arclen = self.poly_max_arclen

        if (poly_num_points is None) and (poly_max_arclen is None):
            raise PatternError('Number of points and arclength left unspecified'
                               ' (default was also overridden)')

        r0, r1 = self.radii

        # Approximate perimeter
        # Ramanujan, S., "Modular Equations and Approximations to ,"
        #  Quart. J. Pure. Appl. Math., vol. 45 (1913-1914), pp. 350-372
        h = ((r1 - r0) / (r1 + r0)) ** 2
        perimeter = pi * (r1 + r0) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

        n = []
        if poly_num_points is not None:
            n += [poly_num_points]
        if poly_max_arclen is not None:
            n += [perimeter / poly_max_arclen]
        thetas = numpy.linspace(2 * pi, 0, max(n), endpoint=False)

        sin_th, cos_th = (numpy.sin(thetas), numpy.cos(thetas))
        xs = r0 * cos_th
        ys = r1 * sin_th
        xys = numpy.vstack((xs, ys)).T

        poly = Polygon(xys, dose=self.dose, layer=self.layer, offset=self.offset)
        poly.rotate(self.rotation)
        return [poly]

    def get_bounds(self) -> numpy.ndarray:
        rot_radii = numpy.dot(rotation_matrix_2d(self.rotation), self.radii)
        return numpy.vstack((self.offset - rot_radii[0],
                             self.offset + rot_radii[1]))

    def rotate(self, theta: float) -> 'Ellipse':
        self.rotation += theta
        return self

    def mirror(self, axis: int) -> 'Ellipse':
        self.offset[axis - 1] *= -1
        self.rotation *= -1
        return self

    def scale_by(self, c: float) -> 'Ellipse':
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
        return (type(self), radii, self.layer), \
               (self.offset, scale/norm_value, angle, self.dose), \
               lambda: Ellipse(radii=radii*norm_value, layer=self.layer)

