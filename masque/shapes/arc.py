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
    The start and stop angle are measured counterclockwise from the first (x) radius.
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
        return self._radii

    @radii.setter
    def radii(self, val: vector2):
        val = numpy.array(val, dtype=float).flatten()
        if not val.size == 2:
            raise PatternError('Radii must have length 2')
        if not val.min() >= 0:
            raise PatternError('Radii must be non-negative')
        self._radii = val

    @property
    def radius_x(self) -> float:
        return self._radii[0]

    @radius_x.setter
    def radius_x(self, val: float):
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self._radii[0] = val

    @property
    def radius_y(self) -> float:
        return self._radii[1]

    @radius_y.setter
    def radius_y(self, val: float):
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self._radii[1] = val

    # arc start/stop angle properties
    @property
    def angles(self) -> vector2:
        """
        Return the start and stop angles [a_start, a_stop].
        Angles are measured from x-axis after rotation

        :return: [a_start, a_stop]
        """
        return self._angles

    @angles.setter
    def angles(self, val: vector2):
        val = numpy.array(val, dtype=float).flatten()
        if not val.size == 2:
            raise PatternError('Angles must have length 2')
        self._angles = val

    @property
    def start_angle(self) -> float:
        return self.angles[0]

    @start_angle.setter
    def start_angle(self, val: float):
        self.angles = (val, self.angles[1])

    @property
    def stop_angle(self) -> float:
        return self.angles[1]

    @stop_angle.setter
    def stop_angle(self, val: float):
        self.angles = (self.angles[0], val)

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
                 width: float,
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
        self.width = width
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

        r0, r1 = self.radii

        # Convert from polar angle to ellipse parameter (for [rx*cos(t), ry*sin(t)] representation)
        a_ranges = self._angles_to_parameters()

        # Approximate perimeter
        # Ramanujan, S., "Modular Equations and Approximations to ,"
        #  Quart. J. Pure. Appl. Math., vol. 45 (1913-1914), pp. 350-372
        a0, a1 = a_ranges[1]    # use outer arc
        h = ((r1 - r0) / (r1 + r0)) ** 2
        ellipse_perimeter = pi * (r1 + r0) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
        perimeter = abs(a0 - a1) / (2 * pi) * ellipse_perimeter         # TODO: make this more accurate

        n = []
        if poly_num_points is not None:
            n += [poly_num_points]
        if poly_max_arclen is not None:
            n += [perimeter / poly_max_arclen]
        thetas_inner = numpy.linspace(a_ranges[0][1], a_ranges[0][0], max(n), endpoint=True)
        thetas_outer = numpy.linspace(a_ranges[1][0], a_ranges[1][1], max(n), endpoint=True)

        sin_th_i, cos_th_i = (numpy.sin(thetas_inner), numpy.cos(thetas_inner))
        sin_th_o, cos_th_o = (numpy.sin(thetas_outer), numpy.cos(thetas_outer))
        wh = self.width / 2.0

        xs1 = (r0 + wh) * cos_th_o
        ys1 = (r1 + wh) * sin_th_o
        xs2 = (r0 - wh) * cos_th_i
        ys2 = (r1 - wh) * sin_th_i

        xs = numpy.hstack((xs1, xs2))
        ys = numpy.hstack((ys1, ys2))
        xys = numpy.vstack((xs, ys)).T

        poly = Polygon(xys, dose=self.dose, layer=self.layer, offset=self.offset)
        poly.rotate(self.rotation)
        return [poly]

    def get_bounds(self) -> numpy.ndarray:
        '''
        Equation for rotated ellipse is
            x = x0 + a * cos(t) * cos(rot) - b * sin(t) * sin(phi)
            y = y0 + a * cos(t) * sin(rot) + b * sin(t) * cos(rot)
          where t is our parameter.

        Differentiating and solving for 0 slope wrt. t, we find
            tan(t) = -+ b/a cot(phi)
          where -+ is for x, y cases, so that's where the extrema are.

        If the extrema are innaccessible due to arc constraints, check the arc endpoints instead.
        '''
        a_ranges = self._angles_to_parameters()

        mins = []
        maxs = []
        for a, sgn in zip(a_ranges, (-1, +1)):
            wh = sgn * self.width/2
            rx = self.radius_x + wh
            ry = self.radius_y + wh

            a0, a1 = a
            a0_offset = a0 - (a0 % (2 * pi))

            sin_r = numpy.sin(self.rotation)
            cos_r = numpy.cos(self.rotation)
            sin_a = numpy.sin(a)
            cos_a = numpy.cos(a)

            # Cutoff angles
            xpt = (-self.rotation) % (2 * pi) + a0_offset
            ypt = (pi/2 - self.rotation) % (2 * pi) + a0_offset
            xnt = (xpt - pi) % (2 * pi) + a0_offset
            ynt = (ypt - pi) % (2 * pi) + a0_offset

            # Points along coordinate axes
            rx2_inv = 1 / (rx * rx)
            ry2_inv = 1 / (ry * ry)
            xr = numpy.abs(cos_r * cos_r * rx2_inv + sin_r * sin_r * ry2_inv) ** -0.5
            yr = numpy.abs(-sin_r * -sin_r * rx2_inv + cos_r * cos_r * ry2_inv) ** -0.5

            # Arc endpoints
            xn, xp = sorted(rx * cos_r * cos_a - ry * sin_r * sin_a)
            yn, yp = sorted(rx * sin_r * cos_a + ry * cos_r * sin_a)

            # If our arc subtends a coordinate axis, use the extremum along that axis
            if a0 < xpt < a1 or a0 < xpt + 2 * pi < a1:
                xp = xr

            if a0 < xnt < a1 or a0 < xnt + 2 * pi < a1:
                xn = -xr

            if a0 < ypt < a1 or a0 < ypt + 2 * pi < a1:
                yp = yr

            if a0 < ynt < a1 or a0 < ynt + 2 * pi < a1:
                yn = -yr

            mins.append([xn, yn])
            maxs.append([xp, yp])
        return numpy.vstack((numpy.min(mins, axis=0) + self.offset,
                             numpy.max(maxs, axis=0) + self.offset))

    def rotate(self, theta: float) -> 'Arc':
        self.rotation += theta
        return self

    def mirror(self, axis: int) -> 'Arc':
        self.offset[axis - 1] *= -1
        self.rotation *= -1
        self.angles *= -1
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

        delta_angle = angles[1] - angles[0]
        start_angle = angles[0] % (2 * pi)
        if start_angle >= pi:
            start_angle -= pi
            rotation += pi

        angles = (start_angle, start_angle + delta_angle)
        rotation %= 2 * pi
        width = self.width

        return (type(self), radii, angles, width, self.layer), \
               (self.offset, scale/norm_value, rotation, self.dose), \
               lambda: Arc(radii=radii*norm_value, angles=angles, width=width, layer=self.layer)

    def get_cap_edges(self) -> numpy.ndarray:
        '''
        :returns: [[[x0, y0], [x1, y1]],   array of 4 points, specifying the two cuts which
                   [[x2, y2], [x3, y3]]],    would create this arc from its corresponding ellipse.
        '''
        a_ranges = self._angles_to_parameters()

        mins = []
        maxs = []
        for a, sgn in zip(a_ranges, (-1, +1)):
            wh = sgn * self.width/2
            rx = self.radius_x + wh
            ry = self.radius_y + wh

            sin_r = numpy.sin(self.rotation)
            cos_r = numpy.cos(self.rotation)
            sin_a = numpy.sin(a)
            cos_a = numpy.cos(a)

            # arc endpoints
            xn, xp = sorted(rx * cos_r * cos_a - ry * sin_r * sin_a)
            yn, yp = sorted(rx * sin_r * cos_a + ry * cos_r * sin_a)

            mins.append([xn, yn])
            maxs.append([xp, yp])
        return  numpy.array([mins, maxs]) + self.offset

    def _angles_to_parameters(self) -> numpy.ndarray:
        '''
        :return: "Eccentric anomaly" parameter ranges for the inner and outer edges, in the form
                   [[a_min_inner, a_max_inner], [a_min_outer, a_max_outer]]
        '''
        a = []
        for sgn in (-1, +1):
            wh = sgn * self.width/2
            rx = self.radius_x + wh
            ry = self.radius_y + wh

            # create paremeter 'a' for parametrized ellipse
            a0, a1 = (numpy.arctan2(rx*numpy.sin(a), ry*numpy.cos(a)) for a in self.angles)
            sign = numpy.sign(self.angles[1] - self.angles[0])
            if sign != numpy.sign(a1 - a0):
                a1 += sign * 2 * pi

            a.append((a0, a1))
        return numpy.array(a)
