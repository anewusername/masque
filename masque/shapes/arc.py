from typing import List
import math
import numpy
from numpy import pi

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_POINTS
from .. import PatternError
from ..utils import is_scalar, vector2


__author__ = 'Jan Petykiewicz'


class Arc(Shape):
    """
    An elliptical arc, formed by cutting off an elliptical ring with two rays which exit from its
     center. It has a position, two radii, a start and stop angle, a rotation, and a width.

    The radii define an ellipse; the ring is formed with radii +/- width/2.
    The rotation gives the angle from x-axis, counterclockwise, to the first (x) radius.
    The start and stop angle are measure counterclockwise from the first (x) radius.
    """

    _radii = None           # type: numpy.ndarray
    _angles = None          # type: numpy.ndarray
    _width = 1.0            # type: float
    _rotation = 0.0         # type: float

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
        return self.radii

    @radii.setter
    def radii(self, val: vector2):
        val = numpy.array(val, dtype=float).flatten()
        if not val.size == 2:
            raise PatternError('Radii must have length 2')
        if not val.min() >= 0:
            raise PatternError('Radii must be non-negative')
        self.radii = val

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

    # arc start/stop angle properties
    @property
    def angles(self) -> vector2:
        """
        Return the start and stop angles [a_start, a_stop].
        Angles are measured from x-axis after rotation, and are stored mod 2*pi

        :return: [a_start, a_stop]
        """
        return self._angles

    @angles.setter
    def angles(self, val: vector2):
        val = numpy.array(val, dtype=float).flatten()
        if not val.size == 2:
            raise PatternError('Angles must have length 2')
        angles = val % (2 * pi)
        if angles[0] > pi:
            self.rotation += pi
            angles -= pi
        self._angles = angles

    @property
    def start_angle(self) -> float:
        return self.angles[0]

    @start_angle.setter
    def start_angle(self, val: float):
        self.angles[0] = val % (2 * pi)

    @property
    def stop_angle(self) -> float:
        return self.angles[1]

    @stop_angle.setter
    def stop_angle(self, val: float):
        self.angles[1] = val % (2 * pi)

    # Rotation property
    @property
    def rotation(self) -> float:
        """
        Rotation of radius_x from x_axis, counterclockwise, in radians. Stored mod 2*pi

        :return: rotation counterclockwise in radians
        """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float):
        if not is_scalar(val):
            raise PatternError('Rotation must be a scalar')
        self._rotation = val % (2 * pi)

    # Width
    @property
    def width(self) -> float:
        """
        Width of the arc (difference between inner and outer radii)

        :return: width
        """
        return self._width

    @width.setter
    def width(self, val: float):
        if not is_scalar(val):
            raise PatternError('Width must be a scalar')
        if not val > 0:
            raise PatternError('Width must be positive')
        self._width = val

    def __init__(self,
                 radii: vector2,
                 angles: vector2,
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
        self.angles = angles
        self.rotation = rotation
        self.poly_num_points = poly_num_points
        self.poly_max_arclen = poly_max_arclen

    def to_polygons(self, poly_num_points: int=None, poly_max_arclen: float=None) -> List[Polygon]:
        if poly_num_points is None:
            poly_num_points = self.poly_num_points
        if poly_max_arclen is None:
            poly_max_arclen = self.poly_max_arclen

        if (poly_num_points is None) and (poly_max_arclen is None):
            raise PatternError('Max number of points and arclength left unspecified' +
                               ' (default was also overridden)')

        rxy = self.radii
        ang = self.angles

        # Approximate perimeter
        # Ramanujan, S., "Modular Equations and Approximations to ,"
        #  Quart. J. Pure. Appl. Math., vol. 45 (1913-1914), pp. 350-372
        h = ((rxy[1] - rxy[0]) / rxy.sum()) ** 2
        ellipse_perimeter = pi * rxy.sum() * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
        perimeter = abs(ang[0] - ang[1]) / (2 * pi) * ellipse_perimeter

        n = []
        if poly_num_points is not None:
            n += [poly_num_points]
        if poly_max_arclen is not None:
            n += [perimeter / poly_max_arclen]
        thetas = numpy.linspace(2 * pi, 0, max(n), endpoint=False)

        sin_th, cos_th = (numpy.sin(thetas), numpy.cos(thetas))
        wh = self.width / 2.0

        xs1 = (rxy[0] + wh) * cos_th - (rxy[1] + wh) * sin_th
        ys1 = (rxy[0] + wh) * cos_th - (rxy[1] + wh) * sin_th
        xs2 = (rxy[0] - wh) * cos_th - (rxy[1] - wh) * sin_th
        ys2 = (rxy[0] - wh) * cos_th - (rxy[1] - wh) * sin_th

        xs = numpy.hstack((xs1, xs2[::-1]))
        ys = numpy.hstack((ys1, ys2[::-1]))
        xys = numpy.vstack((xs, ys)).T

        poly = Polygon(xys, dose=self.dose, layer=self.layer, offset=self.offset)
        poly.rotate(self.rotation)
        return [poly]

    def get_bounds(self) -> numpy.ndarray:
        a = self.angles - 0.5 * pi

        mins = []
        maxs = []
        for sgn in (+1, -1):
            wh = sgn * self.width/2
            rx = self.radius_x + wh
            ry = self.radius_y + wh

            sin_r = numpy.sin(self.rotation)
            cos_r = numpy.cos(self.rotation)
            tan_r = numpy.tan(self.rotation)
            sin_a = numpy.sin(a)
            cos_a = numpy.cos(a)

            xpt = numpy.arctan(-ry / rx * tan_r)
            ypt = numpy.arctan(+ry / rx / tan_r)
            xnt = numpy.arcsin(numpy.sin(xpt - pi))
            ynt = numpy.arcsin(numpy.sin(ypt - pi))

            xr = numpy.sqrt((rx * cos_r) ** 2 + (ry * sin_r) ** 2)
            yr = numpy.sqrt((rx * sin_r) ** 2 + (ry * cos_r) ** 2)

            xn, xp = sorted(rx * cos_r * cos_a - ry * sin_r * sin_a)
            yn, yp = sorted(rx * sin_r * cos_a - ry * cos_r * sin_a)

            if min(a) < xpt < max(a):
                xp = xr

            if min(a) < xnt < max(a):
                xn = -xr

            if min(a) < ypt < max(a):
                yp = yr

            if min(a) < ynt < max(a):
                yn = -yr

            mins.append([xn, yn])
            maxs.append([xp, yp])
        return numpy.vstack((numpy.min(mins, axis=0) + self.offset,
                             numpy.max(maxs, axis=0) + self.offset))

    def rotate(self, theta: float) -> 'Arc':
        self.rotation += theta
        return self

    def scale_by(self, c: float) -> 'Arc':
        self.radii *= c
        self.width *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        if self.radius_x < self.radius_y:
            radii = self.radii / self.radius_x
            scale = self.radius_x
            rotation = self.rotation
            angles = self.angles
        else:  # rotate by 90 degrees and swap radii
            radii = self.radii[::-1] / self.radius_y
            scale = self.radius_y
            rotation = self.rotation + pi / 2
            angles = self.angles - pi / 2
        return (type(self), radii, angles, self.layer), \
               (self.offset, scale/norm_value, rotation, self.dose), \
               lambda: Arc(radii=radii*norm_value, angles=angles, layer=self.layer)


