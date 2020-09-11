from typing import List, Dict, Optional
import copy

import numpy        # type: ignore
from numpy import pi

from . import Shape, Polygon, normalized_shape_tuple, DEFAULT_POLY_NUM_POINTS
from .. import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, vector2, layer_t, AutoSlots, annotations_t
from ..traits import LockableImpl


class Circle(Shape, metaclass=AutoSlots):
    """
    A circle, which has a position and radius.
    """
    __slots__ = ('_radius', 'poly_num_points', 'poly_max_arclen')

    _radius: float
    """ Circle radius """

    poly_num_points: Optional[int]
    """ Sets the default number of points for `.polygonize()` """

    poly_max_arclen: Optional[float]
    """ Sets the default max segement length for `.polygonize()` """

    # radius property
    @property
    def radius(self) -> float:
        """
        Circle's radius (float, >= 0)
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
                 *,
                 poly_num_points: Optional[int] = DEFAULT_POLY_NUM_POINTS,
                 poly_max_arclen: Optional[float] = None,
                 offset: vector2 = (0.0, 0.0),
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
            self._radius = radius
            self._offset = offset
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
            self._layer = layer
            self._dose = dose
        else:
            self.radius = radius
            self.offset = offset
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
            self.layer = layer
            self.dose = dose
        self.poly_num_points = poly_num_points
        self.poly_max_arclen = poly_max_arclen
        self.set_locked(locked)

    def  __deepcopy__(self, memo: Dict = None) -> 'Circle':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new._offset = self._offset.copy()
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
            raise PatternError('Number of points and arclength left '
                               'unspecified (default was also overridden)')

        n = []
        if poly_num_points is not None:
            n += [poly_num_points]
        if poly_max_arclen is not None:
            n += [2 * pi * self.radius / poly_max_arclen]
        num_points = int(round(max(n)))
        thetas = numpy.linspace(2 * pi, 0, num_points, endpoint=False)
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
               (self.offset, magnitude, rotation, False, self.dose), \
               lambda: Circle(radius=norm_value, layer=self.layer)

    def __repr__(self) -> str:
        dose = f' d{self.dose:g}' if self.dose != 1 else ''
        locked = ' L' if self.locked else ''
        return f'<Circle l{self.layer} o{self.offset} r{self.radius:g}{dose}{locked}>'
