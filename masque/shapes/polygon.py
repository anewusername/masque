from typing import Sequence, Any, cast
import copy

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from . import Shape, normalized_shape_tuple
from ..error import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, rotation_matrix_2d
from ..utils import remove_colinear_vertices, remove_duplicate_vertices, annotations_t


class Polygon(Shape):
    """
    A polygon, consisting of a bunch of vertices (Nx2 ndarray) which specify an
       implicitly-closed boundary, and an offset.

    Note that the setter for `Polygon.vertices` may (but may not) create a copy of the
      passed vertex coordinates. See `numpy.array(..., copy=False)` for details.

    A `normalized_form(...)` is available, but can be quite slow with lots of vertices.
    """
    __slots__ = (
        '_vertices',
        # Inherited
        '_offset', '_repetition', '_annotations',
        )

    _vertices: NDArray[numpy.float64]
    """ Nx2 ndarray of vertices `[[x0, y0], [x1, y1], ...]` """

    # vertices property
    @property
    def vertices(self) -> Any:        # mypy#3004   NDArray[numpy.float64]:
        """
        Vertices of the polygon (Nx2 ndarray: `[[x0, y0], [x1, y1], ...]`)

        When setting, note that a copy of the provided vertices may or may not be made,
        following the rules from `numpy.array(.., copy=False)`.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, val: ArrayLike) -> None:
        val = numpy.array(val, dtype=float)
        if len(val.shape) < 2 or val.shape[1] != 2:
            raise PatternError('Vertices must be an Nx2 array')
        if val.shape[0] < 3:
            raise PatternError('Must have at least 3 vertices (Nx2 where N>2)')
        self._vertices = val

    # xs property
    @property
    def xs(self) -> NDArray[numpy.float64]:
        """
        All vertex x coords as a 1D ndarray
        """
        return self.vertices[:, 0]

    @xs.setter
    def xs(self, val: ArrayLike) -> None:
        val = numpy.array(val, dtype=float).flatten()
        if val.size != self.vertices.shape[0]:
            raise PatternError('Wrong number of vertices')
        self.vertices[:, 0] = val

    # ys property
    @property
    def ys(self) -> NDArray[numpy.float64]:
        """
        All vertex y coords as a 1D ndarray
        """
        return self.vertices[:, 1]

    @ys.setter
    def ys(self, val: ArrayLike) -> None:
        val = numpy.array(val, dtype=float).flatten()
        if val.size != self.vertices.shape[0]:
            raise PatternError('Wrong number of vertices')
        self.vertices[:, 1] = val

    def __init__(
            self,
            vertices: ArrayLike,
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(vertices, numpy.ndarray)
            assert isinstance(offset, numpy.ndarray)
            self._vertices = vertices
            self._offset = offset
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
        else:
            self.vertices = vertices
            self.offset = offset
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
        self.rotate(rotation)

    def __deepcopy__(self, memo: dict | None = None) -> 'Polygon':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        new._vertices = self._vertices.copy()
        new._annotations = copy.deepcopy(self._annotations)
        return new

    @staticmethod
    def square(
            side_length: float,
            *,
            rotation: float = 0.0,
            offset: ArrayLike = (0.0, 0.0),
            repetition: Repetition | None = None,
            ) -> 'Polygon':
        """
        Draw a square given side_length, centered on the origin.

        Args:
            side_length: Length of one side
            rotation: Rotation counterclockwise, in radians
            offset: Offset, default `(0, 0)`
            repetition: `Repetition` object, default `None`

        Returns:
            A Polygon object containing the requested square
        """
        norm_square = numpy.array([[-1, -1],
                                   [-1, +1],
                                   [+1, +1],
                                   [+1, -1]], dtype=float)
        vertices = 0.5 * side_length * norm_square
        poly = Polygon(vertices, offset=offset, repetition=repetition)
        poly.rotate(rotation)
        return poly

    @staticmethod
    def rectangle(
            lx: float,
            ly: float,
            *,
            rotation: float = 0,
            offset: ArrayLike = (0.0, 0.0),
            repetition: Repetition | None = None,
            ) -> 'Polygon':
        """
        Draw a rectangle with side lengths lx and ly, centered on the origin.

        Args:
            lx: Length along x (before rotation)
            ly: Length along y (before rotation)
            rotation: Rotation counterclockwise, in radians
            offset: Offset, default `(0, 0)`
            repetition: `Repetition` object, default `None`

        Returns:
            A Polygon object containing the requested rectangle
        """
        vertices = 0.5 * numpy.array([[-lx, -ly],
                                      [-lx, +ly],
                                      [+lx, +ly],
                                      [+lx, -ly]], dtype=float)
        poly = Polygon(vertices, offset=offset, repetition=repetition)
        poly.rotate(rotation)
        return poly

    @staticmethod
    def rect(
            *,
            xmin: float | None = None,
            xctr: float | None = None,
            xmax: float | None = None,
            lx: float | None = None,
            ymin: float | None = None,
            yctr: float | None = None,
            ymax: float | None = None,
            ly: float | None = None,
            repetition: Repetition | None = None,
            ) -> 'Polygon':
        """
        Draw a rectangle by specifying side/center positions.

        Must provide 2 of (xmin, xctr, xmax, lx),
        and 2 of (ymin, yctr, ymax, ly).

        Args:
            xmin: Minimum x coordinate
            xctr: Center x coordinate
            xmax: Maximum x coordinate
            lx: Length along x direction
            ymin: Minimum y coordinate
            yctr: Center y coordinate
            ymax: Maximum y coordinate
            ly: Length along y direction
            repetition: `Repetition` object, default `None`

        Returns:
            A Polygon object containing the requested rectangle
        """
        if lx is None:
            if xctr is None:
                assert xmin is not None
                assert xmax is not None
                xctr = 0.5 * (xmax + xmin)
                lx = xmax - xmin
            elif xmax is None:
                assert xmin is not None
                assert xctr is not None
                lx = 2 * (xctr - xmin)
            elif xmin is None:
                assert xctr is not None
                assert xmax is not None
                lx = 2 * (xmax - xctr)
            else:
                raise PatternError('Two of xmin, xctr, xmax, lx must be None!')
        else:
            if xctr is not None:
                pass
            elif xmax is None:
                assert xmin is not None
                assert lx is not None
                xctr = xmin + 0.5 * lx
            elif xmin is None:
                assert xmax is not None
                assert lx is not None
                xctr = xmax - 0.5 * lx
            else:
                raise PatternError('Two of xmin, xctr, xmax, lx must be None!')

        if ly is None:
            if yctr is None:
                assert ymin is not None
                assert ymax is not None
                yctr = 0.5 * (ymax + ymin)
                ly = ymax - ymin
            elif ymax is None:
                assert ymin is not None
                assert yctr is not None
                ly = 2 * (yctr - ymin)
            elif ymin is None:
                assert yctr is not None
                assert ymax is not None
                ly = 2 * (ymax - yctr)
            else:
                raise PatternError('Two of ymin, yctr, ymax, ly must be None!')
        else:
            if yctr is not None:
                pass
            elif ymax is None:
                assert ymin is not None
                assert ly is not None
                yctr = ymin + 0.5 * ly
            elif ymin is None:
                assert ly is not None
                assert ymax is not None
                yctr = ymax - 0.5 * ly
            else:
                raise PatternError('Two of ymin, yctr, ymax, ly must be None!')

        poly = Polygon.rectangle(lx, ly, offset=(xctr, yctr), repetition=repetition)
        return poly

    @staticmethod
    def octagon(
            *,
            side_length: float | None = None,
            inner_radius: float | None = None,
            regular: bool = True,
            center: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            repetition: Repetition | None = None,
            ) -> 'Polygon':
        """
        Draw an octagon given one of (side length, inradius, circumradius).

        Args:
            side_length: Length of one side. For an irregular octagon, this
                specifies the length of the long sides.
            inner_radius: Half of distance between opposite sides. For an irregular
                octagon, this specifies the spacing between the long sides.
            regular: If `True`, all sides have the same length. If `False`,
                a "clipped square" with vertices (+-1, +-2) and (+-2, +-1)
                is generated, avoiding irrational coordinate locations and
                guaranteeing 45 degree edges.
            center: Offset, default `(0, 0)`
            rotation: Rotation counterclockwise, in radians.
                `0` results in four axis-aligned sides (the long sides of the
                irregular octagon).
            repetition: `Repetition` object, default `None`

        Returns:
            A Polygon object containing the requested octagon
        """
        if regular:
            s = 1 + numpy.sqrt(2)
        else:
            s = 2

        norm_oct = numpy.array([
            [-1, -s],
            [-s, -1],
            [-s,  1],
            [-1,  s],
            [ 1,  s],
            [ s,  1],
            [ s, -1],
            [ 1, -s]], dtype=float)

        if side_length is None:
            if inner_radius is None:
                raise PatternError('One of `side_length` or `inner_radius` must be specified.')
            side_length = 2 * inner_radius / s

        vertices = 0.5 * side_length * norm_oct
        poly = Polygon(vertices, offset=center, repetition=repetition)
        poly.rotate(rotation)
        return poly

    def to_polygons(
            self,
            num_vertices: int | None = None,      # unused
            max_arclen: float | None = None,      # unused
            ) -> list['Polygon']:
        return [copy.deepcopy(self)]

    def get_bounds_single(self) -> NDArray[numpy.float64]:         # TODO note shape get_bounds doesn't include repetition
        return numpy.vstack((self.offset + numpy.min(self.vertices, axis=0),
                             self.offset + numpy.max(self.vertices, axis=0)))

    def rotate(self, theta: float) -> 'Polygon':
        if theta != 0:
            self.vertices = numpy.dot(rotation_matrix_2d(theta), self.vertices.T).T
        return self

    def mirror(self, axis: int = 0) -> 'Polygon':
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
            x_min = cast(Sequence, x_min)[y_min]
        reordered_vertices = numpy.roll(rotated_vertices, -x_min, axis=0)

        # TODO: normalize mirroring?

        return ((type(self), reordered_vertices.data.tobytes()),
                (offset, scale / norm_value, rotation, False),
                lambda: Polygon(reordered_vertices * norm_value))

    def clean_vertices(self) -> 'Polygon':
        """
        Removes duplicate, co-linear and otherwise redundant vertices.

        Returns:
            self
        """
        self.remove_colinear_vertices()
        return self

    def remove_duplicate_vertices(self) -> 'Polygon':
        '''
        Removes all consecutive duplicate (repeated) vertices.

        Returns:
            self
        '''
        self.vertices = remove_duplicate_vertices(self.vertices, closed_path=True)
        return self

    def remove_colinear_vertices(self) -> 'Polygon':
        '''
        Removes consecutive co-linear vertices.

        Returns:
            self
        '''
        self.vertices = remove_colinear_vertices(self.vertices, closed_path=True)
        return self

    def __repr__(self) -> str:
        centroid = self.offset + self.vertices.mean(axis=0)
        return f'<Polygon centroid {centroid} v{len(self.vertices)}>'
