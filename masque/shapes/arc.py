from typing import List, Dict, Optional, Sequence, Any
import copy
import math

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_VERTICES
from ..error import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, layer_t, annotations_t


class Arc(Shape):
    """
    An elliptical arc, formed by cutting off an elliptical ring with two rays which exit from its
     center. It has a position, two radii, a start and stop angle, a rotation, and a width.

    The radii define an ellipse; the ring is formed with radii +/- width/2.
    The rotation gives the angle from x-axis, counterclockwise, to the first (x) radius.
    The start and stop angle are measured counterclockwise from the first (x) radius.
    """
    __slots__ = (
        '_radii', '_angles', '_width', '_rotation',
        # Inherited
        '_offset', '_layer', '_repetition', '_annotations',
        )

    _radii: NDArray[numpy.float64]
    """ Two radii for defining an ellipse """

    _rotation: float
    """ Rotation (ccw, radians) from the x axis to the first radius """

    _angles: NDArray[numpy.float64]
    """ Start and stop angles (ccw, radians) for choosing an arc from the ellipse, measured from the first radius """

    _width: float
    """ Width of the arc """

    # radius properties
    @property
    def radii(self) -> Any:         # TODO mypy#3004   NDArray[numpy.float64]:
        """
        Return the radii `[rx, ry]`
        """
        return self._radii

    @radii.setter
    def radii(self, val: ArrayLike) -> None:
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
    def radius_x(self, val: float) -> None:
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self._radii[0] = val

    @property
    def radius_y(self) -> float:
        return self._radii[1]

    @radius_y.setter
    def radius_y(self, val: float) -> None:
        if not val >= 0:
            raise PatternError('Radius must be non-negative')
        self._radii[1] = val

    # arc start/stop angle properties
    @property
    def angles(self) -> Any:            # TODO mypy#3004    NDArray[numpy.float64]:
        """
        Return the start and stop angles `[a_start, a_stop]`.
        Angles are measured from x-axis after rotation

        Returns:
            `[a_start, a_stop]`
        """
        return self._angles

    @angles.setter
    def angles(self, val: ArrayLike) -> None:
        val = numpy.array(val, dtype=float).flatten()
        if not val.size == 2:
            raise PatternError('Angles must have length 2')
        self._angles = val

    @property
    def start_angle(self) -> float:
        return self.angles[0]

    @start_angle.setter
    def start_angle(self, val: float) -> None:
        self.angles = (val, self.angles[1])

    @property
    def stop_angle(self) -> float:
        return self.angles[1]

    @stop_angle.setter
    def stop_angle(self, val: float) -> None:
        self.angles = (self.angles[0], val)

    # Rotation property
    @property
    def rotation(self) -> float:
        """
        Rotation of radius_x from x_axis, counterclockwise, in radians. Stored mod 2*pi

        Returns:
            rotation counterclockwise in radians
        """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float) -> None:
        if not is_scalar(val):
            raise PatternError('Rotation must be a scalar')
        self._rotation = val % (2 * pi)

    # Width
    @property
    def width(self) -> float:
        """
        Width of the arc (difference between inner and outer radii)

        Returns:
            width
        """
        return self._width

    @width.setter
    def width(self, val: float) -> None:
        if not is_scalar(val):
            raise PatternError('Width must be a scalar')
        if not val > 0:
            raise PatternError('Width must be positive')
        self._width = val

    def __init__(
            self,
            radii: ArrayLike,
            angles: ArrayLike,
            width: float,
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0,
            mirrored: Sequence[bool] = (False, False),
            layer: layer_t = 0,
            repetition: Optional[Repetition] = None,
            annotations: Optional[annotations_t] = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(radii, numpy.ndarray)
            assert isinstance(angles, numpy.ndarray)
            assert isinstance(offset, numpy.ndarray)
            self._radii = radii
            self._angles = angles
            self._width = width
            self._offset = offset
            self._rotation = rotation
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
            self._layer = layer
        else:
            self.radii = radii
            self.angles = angles
            self.width = width
            self.offset = offset
            self.rotation = rotation
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
            self.layer = layer
        [self.mirror(a) for a, do in enumerate(mirrored) if do]

    def __deepcopy__(self, memo: Optional[Dict] = None) -> 'Arc':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        new._radii = self._radii.copy()
        new._angles = self._angles.copy()
        new._annotations = copy.deepcopy(self._annotations)
        return new

    def to_polygons(
            self,
            num_vertices: Optional[int] = DEFAULT_POLY_NUM_VERTICES,
            max_arclen: Optional[float] = None,
            ) -> List[Polygon]:
        if (num_vertices is None) and (max_arclen is None):
            raise PatternError('Max number of points and arclength left unspecified'
                               + ' (default was also overridden)')

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
        if num_vertices is not None:
            n += [num_vertices]
        if max_arclen is not None:
            n += [perimeter / max_arclen]
        num_vertices = int(round(max(n)))

        wh = self.width / 2.0
        if wh == r0 or wh == r1:
            thetas_inner = numpy.zeros(1)      # Don't generate multiple vertices if we're at the origin
        else:
            thetas_inner = numpy.linspace(a_ranges[0][1], a_ranges[0][0], num_vertices, endpoint=True)
        thetas_outer = numpy.linspace(a_ranges[1][0], a_ranges[1][1], num_vertices, endpoint=True)

        sin_th_i, cos_th_i = (numpy.sin(thetas_inner), numpy.cos(thetas_inner))
        sin_th_o, cos_th_o = (numpy.sin(thetas_outer), numpy.cos(thetas_outer))

        xs1 = (r0 + wh) * cos_th_o
        ys1 = (r1 + wh) * sin_th_o
        xs2 = (r0 - wh) * cos_th_i
        ys2 = (r1 - wh) * sin_th_i

        xs = numpy.hstack((xs1, xs2))
        ys = numpy.hstack((ys1, ys2))
        xys = numpy.vstack((xs, ys)).T

        poly = Polygon(xys, layer=self.layer, offset=self.offset, rotation=self.rotation)
        return [poly]

    def get_bounds(self) -> NDArray[numpy.float64]:
        '''
        Equation for rotated ellipse is
            `x = x0 + a * cos(t) * cos(rot) - b * sin(t) * sin(phi)`
            `y = y0 + a * cos(t) * sin(rot) + b * sin(t) * cos(rot)`
          where `t` is our parameter.

        Differentiating and solving for 0 slope wrt. `t`, we find
            `tan(t) = -+ b/a cot(phi)`
          where -+ is for x, y cases, so that's where the extrema are.

        If the extrema are innaccessible due to arc constraints, check the arc endpoints instead.
        '''
        a_ranges = self._angles_to_parameters()

        mins = []
        maxs = []
        for a, sgn in zip(a_ranges, (-1, +1)):
            wh = sgn * self.width / 2
            rx = self.radius_x + wh
            ry = self.radius_y + wh

            if rx == 0 or ry == 0:
                # Single point, at origin
                mins.append([0, 0])
                maxs.append([0, 0])
                continue

            a0, a1 = a
            a0_offset = a0 - (a0 % (2 * pi))

            sin_r = numpy.sin(self.rotation)
            cos_r = numpy.cos(self.rotation)
            sin_a = numpy.sin(a)
            cos_a = numpy.cos(a)

            # Cutoff angles
            xpt = (-self.rotation) % (2 * pi) + a0_offset
            ypt = (pi / 2 - self.rotation) % (2 * pi) + a0_offset
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
        self.rotation += axis * pi
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

        return ((type(self), radii, angles, width / norm_value, self.layer),
                (self.offset, scale / norm_value, rotation, False),
                lambda: Arc(
                    radii=radii * norm_value,
                    angles=angles,
                    width=width * norm_value,
                    layer=self.layer,
                    ))

    def get_cap_edges(self) -> NDArray[numpy.float64]:
        '''
        Returns:
            ```
            [[[x0, y0], [x1, y1]],   array of 4 points, specifying the two cuts which
             [[x2, y2], [x3, y3]]],    would create this arc from its corresponding ellipse.
            ```
        '''
        a_ranges = self._angles_to_parameters()

        mins = []
        maxs = []
        for a, sgn in zip(a_ranges, (-1, +1)):
            wh = sgn * self.width / 2
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
        return numpy.array([mins, maxs]) + self.offset

    def _angles_to_parameters(self) -> NDArray[numpy.float64]:
        '''
        Returns:
            "Eccentric anomaly" parameter ranges for the inner and outer edges, in the form
                   `[[a_min_inner, a_max_inner], [a_min_outer, a_max_outer]]`
        '''
        a = []
        for sgn in (-1, +1):
            wh = sgn * self.width / 2
            rx = self.radius_x + wh
            ry = self.radius_y + wh

            # create paremeter 'a' for parametrized ellipse
            a0, a1 = (numpy.arctan2(rx * numpy.sin(a), ry * numpy.cos(a)) for a in self.angles)
            sign = numpy.sign(self.angles[1] - self.angles[0])
            if sign != numpy.sign(a1 - a0):
                a1 += sign * 2 * pi

            a.append((a0, a1))
        return numpy.array(a)

    def __repr__(self) -> str:
        angles = f' a°{numpy.rad2deg(self.angles)}'
        rotation = f' r°{numpy.rad2deg(self.rotation):g}' if self.rotation != 0 else ''
        return f'<Arc l{self.layer} o{self.offset} r{self.radii}{angles} w{self.width:g}{rotation}>'
