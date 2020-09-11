from typing import List, Tuple, Dict, Sequence, Optional
import copy
import math

import numpy        # type: ignore
from numpy import pi

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_POINTS
from .. import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, rotation_matrix_2d, vector2, layer_t, AutoSlots, annotations_t
from ..traits import LockableImpl


class Ellipse(Shape, metaclass=AutoSlots):
    """
    An ellipse, which has a position, two radii, and a rotation.
    The rotation gives the angle from x-axis, counterclockwise, to the first (x) radius.
    """
    __slots__ = ('_radii', '_rotation',
                 'poly_num_points', 'poly_max_arclen')

    _radii: numpy.ndarray
    """ Ellipse radii """

    _rotation: float
    """ Angle from x-axis to first radius (ccw, radians) """

    poly_num_points: Optional[int]
    """ Sets the default number of points for `.polygonize()` """

    poly_max_arclen: Optional[float]
    """ Sets the default max segement length for `.polygonize()` """

    # radius properties
    @property
    def radii(self) -> numpy.ndarray:
        """
        Return the radii `[rx, ry]`
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

        Returns:
            counterclockwise rotation in radians
        """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float):
        if not is_scalar(val):
            raise PatternError('Rotation must be a scalar')
        self._rotation = val % pi

    def __init__(self,
                 radii: vector2,
                 *,
                 poly_num_points: Optional[int] = DEFAULT_POLY_NUM_POINTS,
                 poly_max_arclen: Optional[float] = None,
                 offset: vector2 = (0.0, 0.0),
                 rotation: float = 0,
                 mirrored: Sequence[bool] = (False, False),
                 layer: layer_t = 0,
                 dose: float = 1.0,
                 repetition: Optional[Repetition] = None,
                 annotations: Optional[annotations_t] = None,
                 locked: bool = False,
                 raw: bool = False,
                 ):
        LockableImpl.unlock(self)
        self.identifier = ()
        if raw:
            self._radii = radii
            self._offset = offset
            self._rotation = rotation
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
            self._layer = layer
            self._dose = dose
        else:
            self.radii = radii
            self.offset = offset
            self.rotation = rotation
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
            self.layer = layer
            self.dose = dose
        [self.mirror(a) for a, do in enumerate(mirrored) if do]
        self.poly_num_points = poly_num_points
        self.poly_max_arclen = poly_max_arclen
        self.set_locked(locked)

    def  __deepcopy__(self, memo: Dict = None) -> 'Ellipse':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new._offset = self._offset.copy()
        new._radii = self._radii.copy()
        new._annotations = copy.deepcopy(self._annotations)
        new.set_locked(self.locked)
        return new

    def to_polygons(self,
                    poly_num_points: Optional[int] = None,
                    poly_max_arclen: Optional[float] = None,
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
        num_points = int(round(max(n)))
        thetas = numpy.linspace(2 * pi, 0, num_points, endpoint=False)

        sin_th, cos_th = (numpy.sin(thetas), numpy.cos(thetas))
        xs = r0 * cos_th
        ys = r1 * sin_th
        xys = numpy.vstack((xs, ys)).T

        poly = Polygon(xys, dose=self.dose, layer=self.layer, offset=self.offset, rotation=self.rotation)
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
               (self.offset, scale/norm_value, angle, False, self.dose), \
               lambda: Ellipse(radii=radii*norm_value, layer=self.layer)

    def lock(self) -> 'Ellipse':
        self.radii.flags.writeable = False
        Shape.lock(self)
        return self

    def unlock(self) -> 'Ellipse':
        Shape.unlock(self)
        self.radii.flags.writeable = True
        return self

    def __repr__(self) -> str:
        rotation = f' r{self.rotation*180/pi:g}' if self.rotation != 0 else ''
        dose = f' d{self.dose:g}' if self.dose != 1 else ''
        locked = ' L' if self.locked else ''
        return f'<Ellipse l{self.layer} o{self.offset} r{self.radii}{rotation}{dose}{locked}>'
