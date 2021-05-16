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
