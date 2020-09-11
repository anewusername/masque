from typing import List, Tuple, Dict, Optional, Sequence
import copy

import numpy        # type: ignore
from numpy import pi

from . import Shape, normalized_shape_tuple
from .. import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, rotation_matrix_2d, vector2, layer_t, AutoSlots
from ..utils import remove_colinear_vertices, remove_duplicate_vertices, annotations_t
from ..traits import LockableImpl


class Polygon(Shape, metaclass=AutoSlots):
    """
    A polygon, consisting of a bunch of vertices (Nx2 ndarray) which specify an
       implicitly-closed boundary, and an offset.

    A `normalized_form(...)` is available, but can be quite slow with lots of vertices.
    """
    __slots__ = ('_vertices',)

    _vertices: numpy.ndarray
    """ Nx2 ndarray of vertices `[[x0, y0], [x1, y1], ...]` """

    # vertices property
    @property
    def vertices(self) -> numpy.ndarray:
        """
        Vertices of the polygon (Nx2 ndarray: `[[x0, y0], [x1, y1], ...]`)
        """
        return self._vertices

    @vertices.setter
    def vertices(self, val: numpy.ndarray):
        val = numpy.array(val, dtype=float)                  #TODO document that these might not be copied
        if len(val.shape) < 2 or val.shape[1] != 2:
            raise PatternError('Vertices must be an Nx2 array')
        if val.shape[0] < 3:
            raise PatternError('Must have at least 3 vertices (Nx2 where N>2)')
        self._vertices = val

    # xs property
    @property
    def xs(self) -> numpy.ndarray:
        """
        All vertex x coords as a 1D ndarray
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
        All vertex y coords as a 1D ndarray
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
                 *,
                 offset: vector2 = (0.0, 0.0),
                 rotation: float = 0.0,
                 mirrored: Sequence[bool] = (False, False),
                 layer: layer_t = 0,
                 dose: float = 1.0,
                 repetition: Optional[Repetition] = None,
                 annotations: Optional[annotations_t] = None,
                 locked: bool = False,
                 raw: bool = False,
                 ):
        LockableImpl.unlock(self)
        self.identifier = ()
        if raw:
            self._vertices = vertices
            self._offset = offset
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
            self._layer = layer
            self._dose = dose
        else:
            self.vertices = vertices
            self.offset = offset
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
            self.layer = layer
            self.dose = dose
        self.rotate(rotation)
        [self.mirror(a) for a, do in enumerate(mirrored) if do]
        self.set_locked(locked)

    def  __deepcopy__(self, memo: Optional[Dict] = None) -> 'Polygon':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new._offset = self._offset.copy()
        new._vertices = self._vertices.copy()
        new._annotations = copy.deepcopy(self._annotations)
        new.set_locked(self.locked)
        return new

    @staticmethod
    def square(side_length: float,
               rotation: float = 0.0,
               offset: vector2 = (0.0, 0.0),
               layer: layer_t = 0,
               dose: float = 1.0,
               ) -> 'Polygon':
        """
        Draw a square given side_length, centered on the origin.

        Args:
            side_length: Length of one side
            rotation: Rotation counterclockwise, in radians
            offset: Offset, default `(0, 0)`
            layer: Layer, default `0`
            dose: Dose, default `1.0`

        Returns:
            A Polygon object containing the requested square
        """
        norm_square = numpy.array([[-1, -1],
                                   [-1, +1],
                                   [+1, +1],
                                   [+1, -1]], dtype=float)
        vertices = 0.5 * side_length * norm_square
        poly = Polygon(vertices, offset=offset, layer=layer, dose=dose)
        poly.rotate(rotation)
        return poly

    @staticmethod
    def rectangle(lx: float,
                  ly: float,
                  rotation: float = 0,
                  offset: vector2 = (0.0, 0.0),
                  layer: layer_t = 0,
                  dose: float = 1.0,
                  ) -> 'Polygon':
        """
        Draw a rectangle with side lengths lx and ly, centered on the origin.

        Args:
            lx: Length along x (before rotation)
            ly: Length along y (before rotation)
            rotation: Rotation counterclockwise, in radians
            offset: Offset, default `(0, 0)`
            layer: Layer, default `0`
            dose: Dose, default `1.0`

        Returns:
            A Polygon object containing the requested rectangle
        """
        vertices = 0.5 * numpy.array([[-lx, -ly],
                                      [-lx, +ly],
                                      [+lx, +ly],
                                      [+lx, -ly]], dtype=float)
        poly = Polygon(vertices, offset=offset, layer=layer, dose=dose)
        poly.rotate(rotation)
        return poly

    @staticmethod
    def rect(xmin: Optional[float] = None,
             xctr: Optional[float] = None,
             xmax: Optional[float] = None,
             lx: Optional[float] = None,
             ymin: Optional[float] = None,
             yctr: Optional[float] = None,
             ymax: Optional[float] = None,
             ly: Optional[float] = None,
             layer: layer_t = 0,
             dose: float = 1.0,
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
            layer: Layer, default `0`
            dose: Dose, default `1.0`

        Returns:
            A Polygon object containing the requested rectangle
        """
        if lx is None:
            if xctr is None:
                assert(xmin is not None)
                assert(xmax is not None)
                xctr = 0.5 * (xmax + xmin)
                lx = xmax - xmin
            elif xmax is None:
                assert(xmin is not None)
                assert(xctr is not None)
                lx = 2 * (xctr - xmin)
            elif xmin is None:
                assert(xctr is not None)
                assert(xmax is not None)
                lx = 2 * (xmax - xctr)
            else:
                raise PatternError('Two of xmin, xctr, xmax, lx must be None!')
        else:
            if xctr is not None:
                pass
            elif xmax is None:
                assert(xmin is not None)
                assert(lx is not None)
                xctr = xmin + 0.5 * lx
            elif xmin is None:
                assert(xmax is not None)
                assert(lx is not None)
                xctr = xmax - 0.5 * lx
            else:
                raise PatternError('Two of xmin, xctr, xmax, lx must be None!')

        if ly is None:
            if yctr is None:
                assert(ymin is not None)
                assert(ymax is not None)
                yctr = 0.5 * (ymax + ymin)
                ly = ymax - ymin
            elif ymax is None:
                assert(ymin is not None)
                assert(yctr is not None)
                ly = 2 * (yctr - ymin)
            elif ymin is None:
                assert(yctr is not None)
                assert(ymax is not None)
                ly = 2 * (ymax - yctr)
            else:
                raise PatternError('Two of ymin, yctr, ymax, ly must be None!')
        else:
            if yctr is not None:
                pass
            elif ymax is None:
                assert(ymin is not None)
                assert(ly is not None)
                yctr = ymin + 0.5 * ly
            elif ymin is None:
                assert(ly is not None)
                assert(ymax is not None)
                yctr = ymax - 0.5 * ly
            else:
                raise PatternError('Two of ymin, yctr, ymax, ly must be None!')

        poly = Polygon.rectangle(lx, ly, offset=(xctr, yctr),
                                 layer=layer, dose=dose)
        return poly


    def to_polygons(self,
                    poly_num_points: int = None,        # unused
                    poly_max_arclen: float = None,      # unused
                    ) -> List['Polygon']:
        return [copy.deepcopy(self)]

    def get_bounds(self) -> numpy.ndarray:
        return numpy.vstack((self.offset + numpy.min(self.vertices, axis=0),
                             self.offset + numpy.max(self.vertices, axis=0)))

    def rotate(self, theta: float) -> 'Polygon':
        if theta != 0:
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

        # TODO: normalize mirroring?

        return (type(self), reordered_vertices.data.tobytes(), self.layer), \
               (offset, scale/norm_value, rotation, False, self.dose), \
               lambda: Polygon(reordered_vertices*norm_value, layer=self.layer)

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

    def lock(self) -> 'Polygon':
        self.vertices.flags.writeable = False
        Shape.lock(self)
        return self

    def unlock(self) -> 'Polygon':
        Shape.unlock(self)
        self.vertices.flags.writeable = True
        return self

    def __repr__(self) -> str:
        centroid = self.offset + self.vertices.mean(axis=0)
        dose = f' d{self.dose:g}' if self.dose != 1 else ''
        locked = ' L' if self.locked else ''
        return f'<Polygon l{self.layer} centroid {centroid} v{len(self.vertices)}{dose}{locked}>'
