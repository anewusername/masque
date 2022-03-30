"""
Vertex list operations
"""
import numpy
from numpy.typing import NDArray, ArrayLike


def remove_duplicate_vertices(vertices: ArrayLike, closed_path: bool = True) -> NDArray[numpy.float64]:
    """
    Given a list of vertices, remove any consecutive duplicates.

    Args:
        vertices: `[[x0, y0], [x1, y1], ...]`
        closed_path: If True, `vertices` is interpreted as an implicity-closed path
            (i.e. the last vertex will be removed if it is the same as the first)

    Returns:
        `vertices` with no consecutive duplicates.
    """
    vertices = numpy.array(vertices)
    duplicates = (vertices == numpy.roll(vertices, 1, axis=0)).all(axis=1)
    if not closed_path:
        duplicates[0] = False
    return vertices[~duplicates]


def remove_colinear_vertices(vertices: ArrayLike, closed_path: bool = True) -> NDArray[numpy.float64]:
    """
    Given a list of vertices, remove any superflous vertices (i.e.
        those which lie along the line formed by their neighbors)

    Args:
        vertices: Nx2 ndarray of vertices
        closed_path: If `True`, the vertices are assumed to represent an implicitly
           closed path. If `False`, the path is assumed to be open. Default `True`.

    Returns:
        `vertices` with colinear (superflous) vertices removed.
    """
    vertices = remove_duplicate_vertices(vertices)

    # Check for dx0/dy0 == dx1/dy1

    dv = numpy.roll(vertices, -1, axis=0) - vertices  # [y1-y0, y2-y1, ...]
    dxdy = dv * numpy.roll(dv, 1, axis=0)[:, ::-1]    # [[dx0*(dy_-1), (dx_-1)*dy0], dx1*dy0, dy1*dx0]]

    dxdy_diff = numpy.abs(numpy.diff(dxdy, axis=1))[:, 0]
    err_mult = 2 * numpy.abs(dxdy).sum(axis=1) + 1e-40

    slopes_equal = (dxdy_diff / err_mult) < 1e-15
    if not closed_path:
        slopes_equal[[0, -1]] = False

    return vertices[~slopes_equal]


def poly_contains_points(
        vertices: ArrayLike,
        points: ArrayLike,
        include_boundary: bool = True,
        ) -> NDArray[numpy.int_]:
    """
    Tests whether the provided points are inside the implicitly closed polygon
    described by the provided list of vertices.

    Args:
        vertices: Nx2 Arraylike of form [[x0, y0], [x1, y1], ...], describing an implicitly-
            closed polygon. Note that this should include any offsets.
        points: Nx2 ArrayLike of form [[x0, y0], [x1, y1], ...] containing the points to test.
        include_boundary: True if points on the boundary should be count as inside the shape.
            Default True.

    Returns:
        ndarray of booleans, [point0_is_in_shape, point1_is_in_shape, ...]
    """
    points = numpy.array(points, copy=False)
    vertices = numpy.array(vertices, copy=False)

    if points.size == 0:
        return numpy.zeros(0)

    min_bounds = numpy.min(vertices, axis=0)[None, :]
    max_bounds = numpy.max(vertices, axis=0)[None, :]

    trivially_outside = ((points < min_bounds).any(axis=1)
                       | (points > max_bounds).any(axis=1))

    nontrivial = ~trivially_outside
    if trivially_outside.all():
        inside = numpy.zeros_like(trivially_outside, dtype=bool)
        return inside

    ntpts = points[None, nontrivial, :]     # nontrivial points, along axis 1 of ndarray
    verts = vertices[:, :, None]

    y0_le = verts[:, 1] <= ntpts[..., 1]      # (axis 0) y_vertex <= y_point (axis 1)
    y1_le = numpy.roll(y0_le, -1, axis=0)          # rolled by 1 vertex

    upward = y0_le & ~y1_le
    downward = ~y0_le & y1_le

    dv = numpy.roll(verts, -1, axis=0) - verts
    is_left = (dv[:, 0] * (ntpts[..., 1] - verts[:, 1])        # >0 if left of dv, <0 if right, 0 if on the line
             - dv[:, 1] * (ntpts[..., 0] - verts[:, 0]))

    winding_number = ((upward & (is_left > 0)).sum(axis=0)
                  - (downward & (is_left < 0)).sum(axis=0))

    nontrivial_inside = winding_number != 0        # filter nontrivial points based on winding number
    if include_boundary:
        nontrivial_inside[(is_left == 0).any(axis=0)] = True        # check if point lies on any edge

    inside = nontrivial.copy()
    inside[nontrivial] = nontrivial_inside
    return inside


