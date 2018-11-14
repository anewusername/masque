from typing import List, Tuple, Callable
from abc import ABCMeta, abstractmethod
import copy
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

    # Layer (integer >= 0 or tuple)
    _layer = 0                          # type: int or Tuple

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
    def mirror(self, axis: int) -> 'Shape':
        """
        Mirror the shape across an axis.

        :param axis: Axis to mirror across.
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
    def copy(self) -> 'Shape':
        """
        Returns a deep copy of the shape.

        :return: Deep copy of self
        """
        return copy.deepcopy(self)

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

    def manhattanize_fast(self, grid_x: numpy.ndarray, grid_y: numpy.ndarray) -> List['Polygon']:
        """
        Returns a list of polygons with grid-aligned ("Manhattan") edges approximating the shape.

        This function works by
            1) Converting the shape to polygons using .to_polygons()
            2) Approximating each edge with an equivalent Manhattan edge
        This process results in a reasonable Manhattan representation of the shape, but is
          imprecise near non-Manhattan or off-grid corners.

        :param grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
        :param grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.
        :return: List of Polygon objects with grid-aligned edges.
        """
        from . import Polygon

        grid_x = numpy.unique(grid_x)
        grid_y = numpy.unique(grid_y)

        polygon_contours = []
        for polygon in self.to_polygons():
            mins, maxs = polygon.get_bounds()

            vertex_lists = []
            p_verts = polygon.vertices + polygon.offset
            for v, v_next in zip(p_verts, numpy.roll(p_verts, -1, axis=0)):
                dv = v_next - v

                if abs(dv[0]) < 1e-20:
                    xs = numpy.array([v[0], v[0]])   # TODO maybe pick between v[0] and v_next[0]?
                    ys = numpy.array([v[1], v_next[1]])
                    xi = numpy.digitize(xs, grid_x).clip(1, len(grid_x) - 1)
                    yi = numpy.digitize(ys, grid_y).clip(1, len(grid_y) - 1)
                    err_x = (xs - grid_x[xi]) / (grid_x[xi] - grid_x[xi - 1])
                    err_y = (ys - grid_y[yi]) / (grid_y[yi] - grid_y[yi - 1])
                    xi[err_y < 0.5] -= 1
                    yi[err_y < 0.5] -= 1

                    segment = numpy.column_stack((grid_x[xi], grid_y[yi]))
                    vertex_lists.append(segment)
                    continue

                m = dv[1]/dv[0]
                def get_grid_inds(xes):
                    ys = m * (xes - v[0]) + v[1]

                    # (inds - 1) is the index of the y-grid line below the edge's intersection with the x-grid
                    inds = numpy.digitize(ys, grid_y).clip(1, len(grid_y) - 1)

                    # err is what fraction of the cell upwards we have to go to reach our y
                    #   (can be negative at bottom edge due to clip above)
                    err = (ys - grid_y[inds - 1]) / (grid_y[inds] - grid_y[inds - 1])

                    # now set inds to the index of the nearest y-grid line
                    inds[err < 0.5] -= 1
                    #if dv[0] >= 0:
                    #    inds[err <= 0.5] -= 1
                    #else:
                    #    inds[err < 0.5] -= 1
                    return inds

                gxi_range = numpy.digitize([v[0], v_next[0]], grid_x)
                gxi_min = numpy.min(gxi_range - 1).clip(0, len(grid_x))
                gxi_max = numpy.max(gxi_range).clip(0, len(grid_x))

                xs = grid_x[gxi_min:gxi_max]
                inds = get_grid_inds(xs)

                # Find intersections for midpoints
                xs2 = (xs[:-1] + xs[1:]) / 2
                inds2 = get_grid_inds(xs2)

                xinds = numpy.round(numpy.arange(gxi_min, gxi_max - 0.99, 1/3)).astype(int)

                # interleave the results
                yinds = xinds.copy()
                yinds[0::3] = inds
                yinds[1::3] = inds2
                yinds[2::3] = inds2

                vlist = numpy.column_stack((grid_x[xinds], grid_y[yinds]))
                if dv[0] < 0:
                    vlist = vlist[::-1]

                vertex_lists.append(vlist)
            polygon_contours.append(numpy.vstack(vertex_lists))

        manhattan_polygons = []
        for contour in polygon_contours:
            manhattan_polygons.append(Polygon(
                vertices=contour,
                layer=self.layer,
                dose=self.dose))

        return manhattan_polygons


    def manhattanize(self, grid_x: numpy.ndarray, grid_y: numpy.ndarray) -> List['Polygon']:
        """
        Returns a list of polygons with grid-aligned ("Manhattan") edges approximating the shape.

        This function works by
            1) Converting the shape to polygons using .to_polygons()
            2) Accurately rasterizing each polygon on a grid,
                where the edges of each grid cell correspond to the allowed coordinates
            3) Thresholding the (anti-aliased) rasterized image
            4) Finding the contours which outline the filled areas in the thresholded image
        This process results in a fairly accurate Manhattan representation of the shape. Possible
          caveats include:
            a) If high accuracy is important, perform any polygonization and clipping operations
                prior to calling this function. This allows you to specify any arguments you may
                need for .to_polygons(), and also avoids calling .manhattanize() multiple times for
                the same grid location (which causes inaccuracies in the final representation).
            b) If the shape is very large or the grid very fine, memory requirements can be reduced
                by breaking the shape apart into multiple, smaller shapes.
            c) Inaccuracies in edge shape can result from Manhattanization of edges which are
                equidistant from allowed edge location.

        Implementation notes:
            i) Rasterization is performed using float_raster, giving a high-precision anti-aliased
                rasterized image.
            ii) To find the exact polygon edges, the thresholded rasterized image is supersampled
                  prior to calling skimage.measure.find_contours(), which uses marching squares
                  to find the contours. This is done because find_contours() performs interpolation,
                  which has to be undone in order to regain the axis-aligned contours. A targetted
                  rewrite of find_contours() for this specific application, or use of a different
                  boundary tracing method could remove this requirement, but for now this seems to
                  be the most performant approach.

        :param grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
        :param grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.
        :return: List of Polygon objects with grid-aligned edges.
        """
        from . import Polygon
        import skimage.measure
        import float_raster

        grid_x = numpy.unique(grid_x)
        grid_y = numpy.unique(grid_y)

        polygon_contours = []
        for polygon in self.to_polygons():
            # Get rid of unused gridlines (anything not within 2 lines of the polygon bounds)
            mins, maxs = polygon.get_bounds()
            keep_x = numpy.logical_and(grid_x > mins[0], grid_x < maxs[0])
            keep_y = numpy.logical_and(grid_y > mins[1], grid_y < maxs[1])
            for k in (keep_x, keep_y):
                for s in (1, 2):
                    k[s:] += k[:-s]
                    k[:-s] += k[s:]
                k = k > 0

            gx = grid_x[keep_x]
            gy = grid_y[keep_y]

            if len(gx) == 0 or len(gy) == 0:
                continue

            offset = (numpy.where(keep_x)[0][0],
                      numpy.where(keep_y)[0][0])

            rastered = float_raster.raster((polygon.vertices + polygon.offset).T, gx, gy)
            binary_rastered = (numpy.abs(rastered) >= 0.5)
            supersampled = binary_rastered.repeat(2, axis=0).repeat(2, axis=1)

            contours = skimage.measure.find_contours(supersampled, 0.5)
            polygon_contours.append((offset, contours))

        manhattan_polygons = []
        for offset_i, contours in polygon_contours:
            for contour in contours:
                # /2 deals with supersampling
                # +.5 deals with the fact that our 0-edge becomes -.5 in the super-sampled contour output
                snapped_contour = numpy.round((contour + .5) / 2).astype(int)
                vertices = numpy.hstack((grid_x[snapped_contour[:, None, 0] + offset_i[0]],
                                         grid_y[snapped_contour[:, None, 1] + offset_i[1]]))

                manhattan_polygons.append(Polygon(
                    vertices=vertices,
                    layer=self.layer,
                    dose=self.dose))

        return manhattan_polygons

