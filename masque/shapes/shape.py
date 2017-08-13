from typing import List, Tuple, Callable
from abc import ABCMeta, abstractmethod
import numpy

from .. import PatternError
from ..utils import is_scalar, rotation_matrix_2d, vector2


__author__ = 'Jan Petykiewicz'


# Type definitions
normalized_shape_tuple = Tuple[Tuple,
                               Tuple[numpy.ndarray, float, float, float],
                               Callable[[], 'Shape']]

# ## Module-wide defaults
# Default number of points per polygon for shapes
DEFAULT_POLY_NUM_POINTS = 24


class Shape(metaclass=ABCMeta):
    """
    Abstract class specifying functions common to all shapes.
    """

    # [x_offset, y_offset]
    _offset = numpy.array([0.0, 0.0])   # type: numpy.ndarray

    # Layer (integer >= 0)
    _layer = 0                          # type: int

    # Dose
    _dose = 1.0                         # type: float

    # --- Abstract methods
    @abstractmethod
    def to_polygons(self, num_vertices: int, max_arclen: float) -> List['Polygon']:
        """
        Returns a list of polygons which approximate the shape.

        :param num_vertices: Number of points to use for each polygon. Can be overridden by
                     max_arclen if that results in more points. Optional, defaults to shapes'
                      internal defaults.
        :param max_arclen: Maximum arclength which can be approximated by a single line
                     segment. Optional, defaults to shapes' internal defaults.
        :return: List of polygons equivalent to the shape
        """
        pass

    @abstractmethod
    def get_bounds(self) -> numpy.ndarray:
        """
        Returns [[x_min, y_min], [x_max, y_max]] which specify a minimal bounding box for the shape.

        :return: [[x_min, y_min], [x_max, y_max]]
        """
        pass

    @abstractmethod
    def rotate(self, theta: float) -> 'Shape':
        """
        Rotate the shape around its center (0, 0), ignoring its offset.

        :param theta: Angle to rotate by (counterclockwise, radians)
        :return: self
        """
        pass

    @abstractmethod
    def scale_by(self, c: float) -> 'Shape':
        """
        Scale the shape's size (eg. radius, for a circle) by a constant factor.

        :param c: Factor to scale by
        :return: self
        """
        pass

    @abstractmethod
    def normalized_form(self, norm_value: int) -> normalized_shape_tuple:
        """
        Writes the shape in a standardized notation, with offset, scale, rotation, and dose
         information separated out from the remaining values.

        :param norm_value: This value is used to normalize lengths intrinsic to teh shape;
                eg. for a circle, the returned magnitude value will be (radius / norm_value), and
                the returned callable will create a Circle(radius=norm_value, ...). This is useful
                when you find it important for quantities to remain in a certain range, eg. for
                GDSII where vertex locations are stored as integers.
        :return: The returned information takes the form of a 3-element tuple,
                (intrinsic, extrinsic, constructor). These are further broken down as:
                extrinsic: ([x_offset, y_offset], scale, rotation, dose)
                intrinsic: A tuple of basic types containing all information about the instance that
                            is not contained in 'extrinsic'. Usually, intrinsic[0] == type(self).
                constructor: A callable (no arguments) which returns an instance of type(self) with
                            internal state equivalent to 'intrinsic'.
        """
        pass

    # ---- Non-abstract properties
    # offset property
    @property
    def offset(self) -> numpy.ndarray:
        """
        [x, y] offset

        :return: [x_offset, y_offset]
        """
        return self._offset

    @offset.setter
    def offset(self, val: vector2):
        if not isinstance(val, numpy.ndarray):
            val = numpy.array(val, dtype=float)

        if val.size != 2:
            raise PatternError('Offset must be convertible to size-2 ndarray')
        self._offset = val.flatten()

    # layer property
    @property
    def layer(self) -> int or Tuple[int]:
        """
        Layer number (int or tuple of ints)

        :return: Layer
        """
        return self._layer

    @layer.setter
    def layer(self, val: int or List[int]):
        self._layer = val

    # dose property
    @property
    def dose(self) -> float:
        """
        Dose (float >= 0)

        :return: Dose value
        """
        return self._dose

    @dose.setter
    def dose(self, val: float):
        if not is_scalar(val):
            raise PatternError('Dose must be a scalar')
        if not val >= 0:
            raise PatternError('Dose must be non-negative')
        self._dose = val

    # ---- Non-abstract methods
    def translate(self, offset: vector2) -> 'Shape':
        """
        Translate the shape by the given offset

        :param offset: [x_offset, y,offset]
        :return: self
        """
        self.offset += offset
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'Shape':
        """
        Rotate the shape around a point.

        :param pivot: Point (x, y) to rotate around
        :param rotation: Angle to rotate by (counterclockwise, radians)
        :return: self
        """
        pivot = numpy.array(pivot, dtype=float)
        self.translate(-pivot)
        self.rotate(rotation)
        self.offset = numpy.dot(rotation_matrix_2d(rotation), self.offset)
        self.translate(+pivot)
        return self

