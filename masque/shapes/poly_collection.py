from typing import Any, cast, Iterable
from collections.abc import Sequence
import copy
import functools

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from . import Shape, normalized_shape_tuple
from ..error import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, rotation_matrix_2d, annotations_lt, annotations_eq, rep2key
from ..utils import remove_colinear_vertices, remove_duplicate_vertices, annotations_t


@functools.total_ordering
class PolyCollection(Shape):
    """
    A collection of polygons, consisting of list of vertex arrays (N_m x 2 ndarrays) which specify
       implicitly-closed boundaries, and an offset.

    Note that the setter for `PolyCollection.vertex_list` creates a copy of the
      passed vertex coordinates.

    A `normalized_form(...)` is available, but can be quite slow with lots of vertices.
    """
    __slots__ = (
        '_vertex_lists',
        # Inherited
        '_offset', '_repetition', '_annotations',
        )

    _vertex_lists: list[NDArray[numpy.float64]]
    """ List of ndarrays (N_m x 2)  of vertices `[ [[x0, y0], [x1, y1], ...] ]` """

    # vertex_lists property
    @property
    def vertex_lists(self) -> Any:        # mypy#3004   NDArray[numpy.float64]:
        """
        Vertices of the polygons (ist of ndarrays (N_m x 2) `[ [[x0, y0], [x1, y1], ...] ]`

        When setting, note that a copy will be made,
        """
        return self._vertex_lists

    @vertex_lists.setter
    def vertex_lists(self, val: ArrayLike) -> None:
        val = [numpy.array(vv, dtype=float) for vv in val]
        for ii, vv in enumerate(val):
            if len(vv.shape) < 2 or vv.shape[1] != 2:
                raise PatternError(f'vertex_lists contents must be an Nx2 arrays (polygon #{ii} fails)')
            if vv.shape[0] < 3:
                raise PatternError(f'vertex_lists contents must have at least 3 vertices (Nx2 where N>2) (polygon ${ii} has shape {vv.shape})')
        self._vertices = val

    # xs property
    @property
    def xs(self) -> NDArray[numpy.float64]:
        """
        All vertex x coords as a 1D ndarray
        """
        return self.vertices[:, 0]

    def __init__(
            self,
            vertex_lists: Iterable[ArrayLike],
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(vertex_lists, list)
            assert all(isinstance(vv, numpy.ndarray) for vv in vertex_lists)
            assert isinstance(offset, numpy.ndarray)
            self._vertex_lists = vertex_lists
            self._offset = offset
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
        else:
            self.vertices = vertices
            self.offset = offset
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
        self.rotate(rotation)

    def __deepcopy__(self, memo: dict | None = None) -> 'PolyCollection':
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        new._vertex_lists = [vv.copy() for vv in self._vertex_lists]
        new._annotations = copy.deepcopy(self._annotations)
        return new

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) is type(other)
            and numpy.array_equal(self.offset, other.offset)
            and all(numpy.array_equal(ss, oo) for ss, oo in zip(self.vertices, other.vertices))
            and self.repetition == other.repetition
            and annotations_eq(self.annotations, other.annotations)
            )

    def __lt__(self, other: Shape) -> bool:
        if type(self) is not type(other):
            if repr(type(self)) != repr(type(other)):
                return repr(type(self)) < repr(type(other))
            return id(type(self)) < id(type(other))

        other = cast(PolyCollection, other)
        for vv, oo in zip(self.vertices, other.vertices):
            if not numpy.array_equal(vv, oo):
                min_len = min(vv.shape[0], oo.shape[0])
                eq_mask = vv[:min_len] != oo[:min_len]
                eq_lt = vv[:min_len] < oo[:min_len]
                eq_lt_masked = eq_lt[eq_mask]
                if eq_lt_masked.size > 0:
                    return eq_lt_masked.flat[0]
                return vv.shape[0] < oo.shape[0]
        if len(self.vertex_lists) != len(other.vertex_lists):
            return len(self.vertex_lists) < len(other.vertex_lists):
        if not numpy.array_equal(self.offset, other.offset):
            return tuple(self.offset) < tuple(other.offset)
        if self.repetition != other.repetition:
            return rep2key(self.repetition) < rep2key(other.repetition)
        return annotations_lt(self.annotations, other.annotations)

    def pop_as_polygon(self, index: int) -> 'Polygon':
        """
        Remove one polygon from the list, and return it as a `Polygon` object.

        Args:
            index: which polygon to pop
        """
        verts = self.vertex_lists.pop(index)
        return Polygon(
            vertices=verts,
            offset=self.offset,
            repetition=self.repetition.copy(),
            annotations=copy.deepcopy(self.annotations),
            )

    def to_polygons(
            self,
            num_vertices: int | None = None,      # unused  # noqa: ARG002
            max_arclen: float | None = None,      # unused  # noqa: ARG002
            ) -> list['Polygon']:
        return [Polygon(
            vertices=vv,
            offset=self.offset,
            repetition=self.repetition.copy(),
            annotations=copy.deepcopy(self.annotations),
            ) for vv in self.vertex_lists]

    def get_bounds_single(self) -> NDArray[numpy.float64]:         # TODO note shape get_bounds doesn't include repetition
        mins = [numpy.min(vv, axis=0) for vv self.vertex_lists]
        maxs = [numpy.max(vv, axis=0) for vv self.vertex_lists]
        return numpy.vstack((self.offset + numpy.min(self.vertex_lists, axis=0),
                             self.offset + numpy.max(self.vertex_lists, axis=0)))

    def rotate(self, theta: float) -> 'Polygon':
        if theta != 0:
            for vv in self.vertex_lists:
                vv[:] = numpy.dot(rotation_matrix_2d(theta), vv.T).T
        return self

    def mirror(self, axis: int = 0) -> 'Polygon':
        for vv in self.vertex_lists:
            vv[:, axis - 1] *= -1
        return self

    def scale_by(self, c: float) -> 'Polygon':
        for vv in self.vertex_lists:
            vv *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        # Note: this function is going to be pretty slow for many-vertexed polygons, relative to
        #   other shapes
        meanv = numpy.concatenate(self.vertex_lists).mean(axis=0)
        zeroed_vertices = [vv - meanv for vv in self.vertex_lists]
        offset = meanv + self.offset

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

    def __repr__(self) -> str:
        centroid = self.offset + numpy.concatenate(self.vertex_lists).mean(axis=0)
        return f'<PolyCollection centroid {centroid} p{len(self.vertex_lists)}>'
