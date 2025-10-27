from typing import Any, cast, Self
from collections.abc import Iterator
import copy
import functools
from itertools import chain

import numpy
from numpy import pi
from numpy.typing import NDArray, ArrayLike

from . import Shape, normalized_shape_tuple
from .polygon import Polygon
from ..error import PatternError
from ..repetition import Repetition
from ..utils import rotation_matrix_2d, annotations_lt, annotations_eq, rep2key, annotations_t


@functools.total_ordering
class PolyCollection(Shape):
    """
    A collection of polygons, consisting of concatenated vertex arrays (N_m x 2 ndarray) which specify
       implicitly-closed boundaries, and an array of offets specifying the first vertex of each
       successive polygon.

    A `normalized_form(...)` is available, but is untested and probably fairly slow.
    """
    __slots__ = (
        '_vertex_lists',
        '_vertex_offsets',
        # Inherited
        '_repetition', '_annotations',
        )

    _vertex_lists: NDArray[numpy.float64]
    """ 2D NDArray ((N+M+...) x 2)  of vertices `[[xa0, ya0], [xa1, ya1], ..., [xb0, yb0], [xb1, yb1], ... ]` """

    _vertex_offsets: NDArray[numpy.intp]
    """ 1D NDArray specifying the starting offset for each polygon """

    @property
    def vertex_lists(self) -> NDArray[numpy.float64]:
        """
        Vertices of the polygons, ((N+M+...) x 2). Use with `vertex_offsets`.
        """
        return self._vertex_lists

    @property
    def vertex_offsets(self) -> NDArray[numpy.intp]:
        """
        Starting offset (in `vertex_lists`) for each polygon
        """
        return self._vertex_offsets

    @property
    def vertex_slices(self) -> Iterator[slice]:
        """
        Iterator which provides slices which index vertex_lists
        """
        for ii, ff in zip(
                self._vertex_offsets,
                chain(self._vertex_offsets, (self._vertex_lists.shape[0],)),
                strict=True,
                ):
            yield slice(ii, ff)

    @property
    def polygon_vertices(self) -> Iterator[NDArray[numpy.float64]]:
        for slc in self.vertex_slices:
            yield self._vertex_lists[slc]

    # Offset property for `Positionable`
    @property
    def offset(self) -> NDArray[numpy.float64]:
        """
        [x, y] offset
        """
        return numpy.zeros(2)

    @offset.setter
    def offset(self, val: ArrayLike) -> None:
        raise PatternError('PolyCollection offset is forced to (0, 0)')

    def set_offset(self, val: ArrayLike) -> Self:
        if numpy.any(val):
            raise PatternError('Path offset is forced to (0, 0)')
        return self

    def translate(self, offset: ArrayLike) -> Self:
        self._vertex_lists += numpy.atleast_2d(offset)
        return self

    def __init__(
            self,
            vertex_lists: ArrayLike,
            vertex_offsets: ArrayLike,
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            repetition: Repetition | None = None,
            annotations: annotations_t = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(vertex_lists, numpy.ndarray)
            assert isinstance(vertex_offsets, numpy.ndarray)
            self._vertex_lists = vertex_lists
            self._vertex_offsets = vertex_offsets
            self._repetition = repetition
            self._annotations = annotations
        else:
            self._vertex_lists = numpy.asarray(vertex_lists, dtype=float)
            self._vertex_offsets = numpy.asarray(vertex_offsets, dtype=numpy.intp)
            self.repetition = repetition
            self.annotations = annotations
        if rotation:
            self.rotate(rotation)
        if numpy.any(offset):
            self.translate(offset)

    def __deepcopy__(self, memo: dict | None = None) -> Self:
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._vertex_lists = self._vertex_lists.copy()
        new._vertex_offsets = self._vertex_offsets.copy()
        new._annotations = copy.deepcopy(self._annotations)
        return new

    def __eq__(self, other: Any) -> bool:
        return (
            type(self) is type(other)
            and numpy.array_equal(self._vertex_lists, other._vertex_lists)
            and numpy.array_equal(self._vertex_offsets, other._vertex_offsets)
            and self.repetition == other.repetition
            and annotations_eq(self.annotations, other.annotations)
            )

    def __lt__(self, other: Shape) -> bool:
        if type(self) is not type(other):
            if repr(type(self)) != repr(type(other)):
                return repr(type(self)) < repr(type(other))
            return id(type(self)) < id(type(other))

        other = cast('PolyCollection', other)

        for vv, oo in zip(self.polygon_vertices, other.polygon_vertices, strict=False):
            if not numpy.array_equal(vv, oo):
                min_len = min(vv.shape[0], oo.shape[0])
                eq_mask = vv[:min_len] != oo[:min_len]
                eq_lt = vv[:min_len] < oo[:min_len]
                eq_lt_masked = eq_lt[eq_mask]
                if eq_lt_masked.size > 0:
                    return eq_lt_masked.flat[0]
                return vv.shape[0] < oo.shape[0]
        if len(self.vertex_lists) != len(other.vertex_lists):
            return len(self.vertex_lists) < len(other.vertex_lists)
        if self.repetition != other.repetition:
            return rep2key(self.repetition) < rep2key(other.repetition)
        return annotations_lt(self.annotations, other.annotations)

    def to_polygons(
            self,
            num_vertices: int | None = None,      # unused  # noqa: ARG002
            max_arclen: float | None = None,      # unused  # noqa: ARG002
            ) -> list['Polygon']:
        return [Polygon(
            vertices = vv,
            repetition = copy.deepcopy(self.repetition),
            annotations = copy.deepcopy(self.annotations),
            ) for vv in self.polygon_vertices]

    def get_bounds_single(self) -> NDArray[numpy.float64]:         # TODO note shape get_bounds doesn't include repetition
        return numpy.vstack((numpy.min(self._vertex_lists, axis=0),
                             numpy.max(self._vertex_lists, axis=0)))

    def rotate(self, theta: float) -> Self:
        if theta != 0:
            rot = rotation_matrix_2d(theta)
            self._vertex_lists = numpy.einsum('ij,kj->ki', rot, self._vertex_lists)
        return self

    def mirror(self, axis: int = 0) -> Self:
        self._vertex_lists[:, axis - 1] *= -1
        return self

    def scale_by(self, c: float) -> Self:
        self._vertex_lists *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        # Note: this function is going to be pretty slow for many-vertexed polygons, relative to
        #   other shapes
        meanv = self._vertex_lists.mean(axis=0)
        zeroed_vertices = self._vertex_lists - [meanv]
        offset = meanv

        scale = zeroed_vertices.std()
        normed_vertices = zeroed_vertices / scale

        _, _, vertex_axis = numpy.linalg.svd(zeroed_vertices)
        rotation = numpy.arctan2(vertex_axis[0][1], vertex_axis[0][0]) % (2 * pi)
        rotated_vertices = numpy.einsum('ij,kj->ki', rotation_matrix_2d(-rotation), normed_vertices)

        # TODO consider how to reorder vertices for polycollection
        ## Reorder the vertices so that the one with lowest x, then y, comes first.
        #x_min = rotated_vertices[:, 0].argmin()
        #if not is_scalar(x_min):
        #    y_min = rotated_vertices[x_min, 1].argmin()
        #    x_min = cast('Sequence', x_min)[y_min]
        #reordered_vertices = numpy.roll(rotated_vertices, -x_min, axis=0)

        # TODO: normalize mirroring?

        return ((type(self), rotated_vertices.data.tobytes() + self._vertex_offsets.tobytes()),
                (offset, scale / norm_value, rotation, False),
                lambda: PolyCollection(
                    vertex_lists=rotated_vertices * norm_value,
                    vertex_offsets=self._vertex_offsets,
                    ),
                )

    def __repr__(self) -> str:
        centroid = self.vertex_lists.mean(axis=0)
        return f'<PolyCollection centroid {centroid} p{len(self.vertex_offsets)}>'
