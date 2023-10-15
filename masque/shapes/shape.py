from typing import Callable, Self, TYPE_CHECKING
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import NDArray, ArrayLike

from ..traits import (
    Rotatable, Mirrorable, Copyable, Scalable,
    PositionableImpl, PivotableImpl, RepeatableImpl, AnnotatableImpl,
    )

if TYPE_CHECKING:
    from . import Polygon


# Type definitions
normalized_shape_tuple = tuple[
    tuple,
    tuple[NDArray[numpy.float64], float, float, bool],
    Callable[[], 'Shape'],
    ]

# ## Module-wide defaults
# Default number of points per polygon for shapes
DEFAULT_POLY_NUM_VERTICES = 24


class Shape(PositionableImpl, Rotatable, Mirrorable, Copyable, Scalable,
            PivotableImpl, RepeatableImpl, AnnotatableImpl, metaclass=ABCMeta):
    """
    Class specifying functions common to all shapes.
    """
    __slots__ = ()      # Children should use AutoSlots or set slots themselves

    def __copy__(self) -> Self:
        cls = self.__class__
        new = cls.__new__(cls)
        for name in self.__slots__:     # type: str
            object.__setattr__(new, name, getattr(self, name))
        return new

    #
    # Methods (abstract)
    #
    @abstractmethod
    def to_polygons(
            self,
            num_vertices: int | None = None,
            max_arclen: float | None = None,
            ) -> list['Polygon']:
        """
        Returns a list of polygons which approximate the shape.

        Args:
            num_vertices: Number of points to use for each polygon. Can be overridden by
                  max_arclen if that results in more points. Optional, defaults to shapes'
                  internal defaults.
            max_arclen: Maximum arclength which can be approximated by a single line
                  segment. Optional, defaults to shapes' internal defaults.

        Returns:
            List of polygons equivalent to the shape
        """
        pass

    @abstractmethod
    def normalized_form(self, norm_value: int) -> normalized_shape_tuple:
        """
        Writes the shape in a standardized notation, with offset, scale, and rotation
         information separated out from the remaining values.

        Args:
            norm_value: This value is used to normalize lengths intrinsic to the shape;
                eg. for a circle, the returned intrinsic radius value will be (radius / norm_value), and
                the returned callable will create a `Circle(radius=norm_value, ...)`. This is useful
                when you find it important for quantities to remain in a certain range, eg. for
                GDSII where vertex locations are stored as integers.

        Returns:
            The returned information takes the form of a 3-element tuple,
              `(intrinsic, extrinsic, constructor)`. These are further broken down as:
              `intrinsic`: A tuple of basic types containing all information about the instance that
                         is not contained in 'extrinsic'. Usually, `intrinsic[0] == type(self)`.
              `extrinsic`: `([x_offset, y_offset], scale, rotation, mirror_across_x_axis)`
              `constructor`: A callable (no arguments) which returns an instance of `type(self)` with
                           internal state equivalent to `intrinsic`.
        """
        pass

    #
    # Non-abstract methods
    #
    def manhattanize_fast(
            self,
            grid_x: ArrayLike,
            grid_y: ArrayLike,
            ) -> list['Polygon']:
        """
        Returns a list of polygons with grid-aligned ("Manhattan") edges approximating the shape.

        This function works by
            1) Converting the shape to polygons using `.to_polygons()`
            2) Approximating each edge with an equivalent Manhattan edge
        This process results in a reasonable Manhattan representation of the shape, but is
          imprecise near non-Manhattan or off-grid corners.

        Args:
            grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
            grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.

        Returns:
            List of `Polygon` objects with grid-aligned edges.
        """
        from . import Polygon

        gx = numpy.unique(grid_x)
        gy = numpy.unique(grid_y)

        polygon_contours = []
        for polygon in self.to_polygons():
            bounds = polygon.get_bounds_single()
            if bounds is None:
                continue

            mins, maxs = bounds

            vertex_lists = []
            p_verts = polygon.vertices + polygon.offset
            for v, v_next in zip(p_verts, numpy.roll(p_verts, -1, axis=0)):
                dv = v_next - v

                # Find x-index bounds for the line      # TODO: fix this and err_xmin/xmax for grids smaller than the line / shape
                gxi_range = numpy.digitize([v[0], v_next[0]], gx)
                gxi_min = numpy.min(gxi_range - 1).clip(0, len(gx) - 1)
                gxi_max = numpy.max(gxi_range).clip(0, len(gx))

                err_xmin = (min(v[0], v_next[0]) - gx[gxi_min]) / (gx[gxi_min + 1] - gx[gxi_min])
                err_xmax = (max(v[0], v_next[0]) - gx[gxi_max - 1]) / (gx[gxi_max] - gx[gxi_max - 1])

                if err_xmin >= 0.5:
                    gxi_min += 1
                if err_xmax >= 0.5:
                    gxi_max += 1

                if abs(dv[0]) < 1e-20:
                    # Vertical line, don't calculate slope
                    xi = [gxi_min, gxi_max - 1]
                    ys = numpy.array([v[1], v_next[1]])
                    yi = numpy.digitize(ys, gy).clip(1, len(gy) - 1)
                    err_y = (ys - gy[yi]) / (gy[yi] - gy[yi - 1])
                    yi[err_y < 0.5] -= 1

                    segment = numpy.column_stack((gx[xi], gy[yi]))
                    vertex_lists.append(segment)
                    continue

                m = dv[1] / dv[0]

                def get_grid_inds(xes: ArrayLike) -> NDArray[numpy.float64]:
                    ys = m * (xes - v[0]) + v[1]

                    # (inds - 1) is the index of the y-grid line below the edge's intersection with the x-grid
                    inds = numpy.digitize(ys, gy).clip(1, len(gy) - 1)

                    # err is what fraction of the cell upwards we have to go to reach our y
                    #   (can be negative at bottom edge due to clip above)
                    err = (ys - gy[inds - 1]) / (gy[inds] - gy[inds - 1])

                    # now set inds to the index of the nearest y-grid line
                    inds[err < 0.5] -= 1
                    return inds

                # Find the y indices on all x gridlines
                xs = gx[int(gxi_min):int(gxi_max)]
                inds = get_grid_inds(xs)

                # Find y-intersections for x-midpoints
                xs2 = (xs[:-1] + xs[1:]) / 2
                inds2 = get_grid_inds(xs2)

                xinds = numpy.rint(numpy.arange(gxi_min, gxi_max - 0.99, 1 / 3)).astype(numpy.int64)

                # interleave the results
                yinds = xinds.copy()
                yinds[0::3] = inds
                yinds[1::3] = inds2
                yinds[2::3] = inds2

                vlist = numpy.column_stack((gx[xinds], gy[yinds]))
                if dv[0] < 0:
                    vlist = vlist[::-1]

                vertex_lists.append(vlist)
            polygon_contours.append(numpy.vstack(vertex_lists))

        manhattan_polygons = [Polygon(vertices=contour) for contour in polygon_contours]

        return manhattan_polygons

    def manhattanize(
            self,
            grid_x: ArrayLike,
            grid_y: ArrayLike,
            ) -> list['Polygon']:
        """
        Returns a list of polygons with grid-aligned ("Manhattan") edges approximating the shape.

        This function works by
            1) Converting the shape to polygons using `.to_polygons()`
            2) Accurately rasterizing each polygon on a grid,
                where the edges of each grid cell correspond to the allowed coordinates
            3) Thresholding the (anti-aliased) rasterized image
            4) Finding the contours which outline the filled areas in the thresholded image
        This process results in a fairly accurate Manhattan representation of the shape. Possible
          caveats include:
            a) If high accuracy is important, perform any polygonization and clipping operations
                prior to calling this function. This allows you to specify any arguments you may
                need for `.to_polygons()`, and also avoids calling `.manhattanize()` multiple times for
                the same grid location (which causes inaccuracies in the final representation).
            b) If the shape is very large or the grid very fine, memory requirements can be reduced
                by breaking the shape apart into multiple, smaller shapes.
            c) Inaccuracies in edge shape can result from Manhattanization of edges which are
                equidistant from allowed edge location.

        Implementation notes:
            i) Rasterization is performed using `float_raster`, giving a high-precision anti-aliased
                rasterized image.
            ii) To find the exact polygon edges, the thresholded rasterized image is supersampled
                  prior to calling `skimage.measure.find_contours()`, which uses marching squares
                  to find the contours. This is done because `find_contours()` performs interpolation,
                  which has to be undone in order to regain the axis-aligned contours. A targetted
                  rewrite of `find_contours()` for this specific application, or use of a different
                  boundary tracing method could remove this requirement, but for now this seems to
                  be the most performant approach.

        Args:
            grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
            grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.

        Returns:
            List of `Polygon` objects with grid-aligned edges.
        """
        from . import Polygon
        import skimage.measure      # type: ignore
        import float_raster

        grx = numpy.unique(grid_x)
        gry = numpy.unique(grid_y)

        polygon_contours = []
        for polygon in self.to_polygons():
            # Get rid of unused gridlines (anything not within 2 lines of the polygon bounds)
            bounds = polygon.get_bounds_single()
            if bounds is None:
                continue

            mins, maxs = bounds
            keep_x = numpy.logical_and(grx > mins[0], grx < maxs[0])
            keep_y = numpy.logical_and(gry > mins[1], gry < maxs[1])
            for k in (keep_x, keep_y):
                for s in (1, 2):
                    k[s:] += k[:-s]
                    k[:-s] += k[s:]
                k = k > 0

            gx = grx[keep_x]
            gy = gry[keep_y]

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
                snapped_contour = numpy.rint((contour + .5) / 2).astype(numpy.int64)
                vertices = numpy.hstack((grx[snapped_contour[:, None, 0] + offset_i[0]],
                                         gry[snapped_contour[:, None, 1] + offset_i[1]]))

                manhattan_polygons.append(Polygon(vertices=vertices))

        return manhattan_polygons
