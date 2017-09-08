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
        Draw a square given side_length

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
        Draw a rectangle with side lengths lx and ly

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

    def cut(self,
            cut_xs: numpy.ndarray = None,
            cut_ys: numpy.ndarray = None
            ) -> List['Polygon']:
        """
        Decomposes the polygon into a list of constituents by cutting along the
          specified x and/or y coordinates.

        :param cut_xs: list of x-coordinates to cut along (e.g., [1, 1.4, 6])
        :param cut_ys: list of y-coordinates to cut along (e.g., [1, 3, 5.4])
        :return: List of Polygon objects
        """
        import float_raster
        xy_complex = self.vertices[:, 0] + 1j * self.vertices[:, 1]
        xy_cleaned = _clean_complex_vertices(xy_complex)
        xy = numpy.vstack((numpy.real(xy_cleaned)[None, :],
                           numpy.imag(xy_cleaned)[None, :]))

        if cut_xs is None:
            cut_xs = tuple()
        if cut_ys is None:
            cut_ys = tuple()

        mins, maxs = self.get_bounds()
        dx, dy = maxs - mins

        cx = numpy.hstack((min(tuple(cut_xs) + (mins[0],)) - dx, cut_xs, max((maxs[0],) + tuple(cut_xs)) + dx))
        cy = numpy.hstack((min(tuple(cut_ys) + (mins[1],)) - dy, cut_ys, max((maxs[1],) + tuple(cut_ys)) + dy))

        all_verts = float_raster.create_vertices(xy, cx, cy)

        polygons = []
        for cx_min, cx_max in zip(cx, cx[1:]):
            for cy_min, cy_max in zip(cy, cy[1:]):
                clipped_verts = (numpy.real(all_verts).clip(cx_min, cx_max) + 1j *
                                 numpy.imag(all_verts).clip(cy_min, cy_max))

                cleaned_verts = _clean_complex_vertices(clipped_verts)
                if len(cleaned_verts) == 0:
                    continue

                final_verts = numpy.hstack((numpy.real(cleaned_verts)[:, None],
                                            numpy.imag(cleaned_verts)[:, None]))
                polygons.append(Polygon(
                    vertices=final_verts,
                    layer=self.layer,
                    dose=self.dose))
        return polygons


def _clean_complex_vertices(vertices: numpy.ndarray) -> numpy.ndarray:
    eps = numpy.finfo(vertices.dtype).eps

    def cleanup(v):
        # Remove duplicate points
        dv = v - numpy.roll(v, 1)
        v = v[numpy.abs(dv) > eps]

        # Remove colinear points
        dv = v - numpy.roll(v, 1)
        m = numpy.angle(dv) % pi
        diff_m = m - numpy.roll(m, -1)
        return v[numpy.abs(diff_m) > eps]

    n = len(vertices)
    cleaned = cleanup(vertices)
    while n != len(cleaned):
        n = len(cleaned)
        cleaned = cleanup(cleaned)

    return cleaned

