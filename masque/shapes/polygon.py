from typing import List
import copy
import numpy
from numpy import pi

from . import Shape, normalized_shape_tuple
from .. import PatternError
from ..utils import is_scalar, rotation_matrix_2d, vector2

__author__ = 'Jan Petykiewicz'


class Polygon(Shape):
    """
    A polygon, consisting of a bunch of vertices (Nx2 ndarray) along with an offset.

    A normalized_form(...) is available, but can be quite slow with lots of vertices.
    """
    _vertices = None        # type: numpy.ndarray

    # vertices property
    @property
    def vertices(self) -> numpy.ndarray:
        """
        Vertices of the polygon (Nx2 ndarray: [[x0, y0], [x1, y1], ...]

        :return: vertices
        """
        return self._vertices

    @vertices.setter
    def vertices(self, val: numpy.ndarray):
        val = numpy.array(val, dtype=float)
        if len(val.shape) < 2 or val.shape[1] != 2:
            raise PatternError('Vertices must be an Nx2 array')
        if val.shape[0] < 3:
            raise PatternError('Must have at least 3 vertices (Nx2, N>3)')
        self._vertices = val

    # xs property
    @property
    def xs(self) -> numpy.ndarray:
        """
        All x vertices in a 1D ndarray
        """
        return self.vertices[:, 0]

    @xs.setter
    def xs(self, val: numpy.ndarray):
        val = numpy.array(val, dtype=float).flatten()
        if val.size != self.vertices.shape[0]:
            raise PatternError('Wrong number of vertices')
        self.vertices[:, 0] = val

    # ys property
    @property
    def ys(self) -> numpy.ndarray:
        """
        All y vertices in a 1D ndarray
        """
        return self.vertices[:, 1]

    @ys.setter
    def ys(self, val: numpy.ndarray):
        val = numpy.array(val, dtype=float).flatten()
        if val.size != self.vertices.shape[0]:
            raise PatternError('Wrong number of vertices')
        self.vertices[:, 1] = val

    def __init__(self,
                 vertices: numpy.ndarray,
                 offset: vector2=(0.0, 0.0),
                 layer: int=0,
                 dose: float=1.0):
        self.offset = offset
        self.layer = layer
        self.dose = dose
        self.vertices = vertices

    @staticmethod
    def square(side_length: float,
               rotation: float=0.0,
               offset: vector2=(0.0, 0.0),
               layer: int=0,
               dose: float=1.0
               ) -> 'Polygon':
        """
        Draw a square given side_length, centered on the origin.

        :param side_length: Length of one side
        :param rotation: Rotation counterclockwise, in radians
        :param offset: Offset, default (0, 0)
        :param layer: Layer, default 0
        :param dose: Dose, default 1.0
        :return: A Polygon object containing the requested square
        """
        norm_square = numpy.array([[-1, -1],
                                   [-1, +1],
                                   [+1, +1],
                                   [+1, -1]], dtype=float)
        vertices = 0.5 * side_length * norm_square
        poly = Polygon(vertices, offset, layer, dose)
        poly.rotate(rotation)
        return poly

    @staticmethod
    def rectangle(lx: float,
                  ly: float,
                  rotation: float=0,
                  offset: vector2=(0.0, 0.0),
                  layer: int=0,
                  dose: float=1.0
                  ) -> 'Polygon':
        """
        Draw a rectangle with side lengths lx and ly, centered on the origin.

        :param lx: Length along x (before rotation)
        :param ly: Length along y (before rotation)
        :param rotation: Rotation counterclockwise, in radians
        :param offset: Offset, default (0, 0)
        :param layer: Layer, default 0
        :param dose: Dose, default 1.0
        :return: A Polygon object containing the requested rectangle
        """
        vertices = 0.5 * numpy.array([[-lx, -ly],
                                      [-lx, +ly],
                                      [+lx, +ly],
                                      [+lx, -ly]], dtype=float)
        poly = Polygon(vertices, offset, layer, dose)
        poly.rotate(rotation)
        return poly

    @staticmethod
    def rect(xmin: float = None,
             xctr: float = None,
             xmax: float = None,
             lx: float = None,
             ymin: float = None,
             yctr: float = None,
             ymax: float = None,
             ly: float = None,
             layer: int = 0,
             dose: float = 1.0
             ) -> 'Polygon':
        """
        Draw a rectangle by specifying side/center positions.

        Must provide 2 of (xmin, xctr, xmax, lx),
        and 2 of (ymin, yctr, ymax, ly).

        :param xmin: Minimum x coordinate
        :param xctr: Center x coordinate
        :param xmax: Maximum x coordinate
        :param lx: Length along x direction
        :param ymin: Minimum y coordinate
        :param yctr: Center y coordinate
        :param ymax: Maximum y coordinate
        :param ly: Length along y direction
        :param layer: Layer, default 0
        :param dose: Dose, default 1.0
        :return: A Polygon object containing the requested rectangle
        """
        if lx is None:
            if xctr is None:
                xctr = 0.5 * (xmax + xmin)
                lx = xmax - xmin
            elif xmax is None:
                lx = 2 * (xctr - xmin)
            elif xmin is None:
                lx = 2 * (xmax - xctr)
            else:
                raise PatternError('Two of xmin, xctr, xmax, lx must be None!')
        else:
            if xctr is not None:
                pass
            elif xmax is None:
                xctr = xmin + 0.5 * lx
            elif xmin is None:
                xctr = xmax - 0.5 * lx
            else:
                raise PatternError('Two of xmin, xctr, xmax, lx must be None!')

        if ly is None:
            if yctr is None:
                yctr = 0.5 * (ymax + ymin)
                ly = ymax - ymin
            elif ymax is None:
                ly = 2 * (yctr - ymin)
            elif ymin is None:
                ly = 2 * (ymax - yctr)
            else:
                raise PatternError('Two of ymin, yctr, ymax, ly must be None!')
        else:
            if yctr is not None:
                pass
            elif ymax is None:
                yctr = ymin + 0.5 * ly
            elif ymin is None:
                yctr = ymax - 0.5 * ly
            else:
                raise PatternError('Two of ymin, yctr, ymax, ly must be None!')

        poly = Polygon.rectangle(lx, ly, offset=(xctr, yctr),
                                 layer=layer, dose=dose)
        return poly


    def to_polygons(self,
                    _poly_num_points: int=None,
                    _poly_max_arclen: float=None,
                    ) -> List['Polygon']:
        return [copy.deepcopy(self)]

    def get_bounds(self) -> numpy.ndarray:
        return numpy.vstack((self.offset + numpy.min(self.vertices, axis=0),
                             self.offset + numpy.max(self.vertices, axis=0)))

    def rotate(self, theta: float) -> 'Polygon':
        self.vertices = numpy.dot(rotation_matrix_2d(theta), self.vertices.T).T
        return self

    def mirror(self, axis: int) -> 'Polygon':
        self.vertices[:, axis - 1] *= -1
        return self

    def scale_by(self, c: float) -> 'Polygon':
        self.vertices *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        # Note: this function is going to be pretty slow for many-vertexed polygons, relative to
        #   other shapes
        offset = self.vertices.mean(axis=0) + self.offset
        zeroed_vertices = self.vertices - offset

        scale = zeroed_vertices.std()
        normed_vertices = zeroed_vertices / scale

        _, _, vertex_axis = numpy.linalg.svd(zeroed_vertices)
        rotation = numpy.arctan2(vertex_axis[0][1], vertex_axis[0][0]) % (2 * pi)
        rotated_vertices = numpy.vstack([numpy.dot(rotation_matrix_2d(-rotation), v)
                                        for v in normed_vertices])

        # Reorder the vertices so that the one with lowest x, then y, comes first.
        x_min = rotated_vertices[:, 0].argmin()
        if not is_scalar(x_min):
            y_min = rotated_vertices[x_min, 1].argmin()
            x_min = x_min[y_min]
        reordered_vertices = numpy.roll(rotated_vertices, -x_min, axis=0)

        return (type(self), reordered_vertices.data.tobytes(), self.layer), \
               (offset, scale/norm_value, rotation, self.dose), \
               lambda: Polygon(reordered_vertices*norm_value, layer=self.layer)

    def clean_vertices(self) -> 'Polygon':
        """
        Removes duplicate, co-linear and otherwise redundant vertices.

        :returns: self
        """
        self.remove_colinear_vertices()
        return self

    def remove_duplicate_vertices(self) -> 'Polygon':
        '''
        Removes all consecutive duplicate (repeated) vertices.

        :returns: self
        '''
        duplicates = (self.vertices == numpy.roll(self.vertices, 1, axis=0)).all(axis=1)
        self.vertices = self.vertices[~duplicates]
        return self

    def remove_colinear_vertices(self) -> 'Polygon':
        '''
        Removes consecutive co-linear vertices.

        :returns: self
        '''
        dv0 = numpy.roll(self.vertices, 1, axis=0) - self.vertices
        dv1 = numpy.roll(dv0, -1, axis=0)

        # find cases where at least one coordinate is 0 in successive dv's
        eq = dv1 == dv0
        aa_colinear = numpy.logical_and(eq, dv0 == 0).any(axis=1)

        # find cases where slope is equal
        with numpy.errstate(divide='ignore', invalid='ignore'):   # don't care about zeroes
            slope_quotient = (dv0[:, 0] * dv1[:, 1]) / (dv1[:, 0] * dv0[:, 1])
        slopes_equal = numpy.abs(slope_quotient - 1) < 1e-14

        colinear = numpy.logical_or(aa_colinear, slopes_equal)
        self.vertices = self.vertices[~colinear]
        return self

