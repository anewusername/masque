from typing import List, Tuple, Dict, Optional, Sequence
import copy
from enum import Enum

import numpy        # type: ignore
from numpy import pi, inf

from . import Shape, normalized_shape_tuple, Polygon, Circle
from .. import PatternError
from ..repetition import Repetition
from ..utils import is_scalar, rotation_matrix_2d, vector2, layer_t, AutoSlots
from ..utils import remove_colinear_vertices, remove_duplicate_vertices, annotations_t
from ..traits import LockableImpl


class PathCap(Enum):
    Flush = 0       # Path ends at final vertices
    Circle = 1      # Path extends past final vertices with a semicircle of radius width/2
    Square = 2      # Path extends past final vertices with a width-by-width/2 rectangle
    SquareCustom = 4  # Path extends past final vertices with a rectangle of length
                      #     defined by path.cap_extensions


class Path(Shape, metaclass=AutoSlots):
    """
    A path, consisting of a bunch of vertices (Nx2 ndarray), a width, an end-cap shape,
        and an offset.

    A normalized_form(...) is available, but can be quite slow with lots of vertices.
    """
    __slots__ = ('_vertices', '_width', '_cap', '_cap_extensions')
    _vertices: numpy.ndarray
    _width: float
    _cap: PathCap
    _cap_extensions: Optional[numpy.ndarray]

    Cap = PathCap

    # width property
    @property
    def width(self) -> float:
        """
        Path width (float, >= 0)
        """
        return self._width

    @width.setter
    def width(self, val: float):
        if not is_scalar(val):
            raise PatternError('Width must be a scalar')
        if not val >= 0:
            raise PatternError('Width must be non-negative')
        self._width = val

    # cap property
    @property
    def cap(self) -> PathCap:
        """
        Path end-cap
        """
        return self._cap

    @cap.setter
    def cap(self, val: PathCap):
        # TODO: Document that setting cap can change cap_extensions
        self._cap = PathCap(val)
        if self.cap != PathCap.SquareCustom:
            self.cap_extensions = None
        elif self.cap_extensions is None:
            # just got set to SquareCustom
            self.cap_extensions = numpy.zeros(2)

    # cap_extensions property
    @property
    def cap_extensions(self) -> Optional[numpy.ndarray]:
        """
        Path end-cap extension

        Returns:
            2-element ndarray or `None`
        """
        return self._cap_extensions

    @cap_extensions.setter
    def cap_extensions(self, vals: Optional[numpy.ndarray]):
        custom_caps = (PathCap.SquareCustom,)
        if self.cap in custom_caps:
            if vals is None:
                raise Exception('Tried to set cap extensions to None on path with custom cap type')
            self._cap_extensions = numpy.array(vals, dtype=float)
        else:
            if vals is not None:
                raise Exception('Tried to set custom cap extensions on path with non-custom cap type')
            self._cap_extensions = vals

    # vertices property
    @property
    def vertices(self) -> numpy.ndarray:
        """
        Vertices of the path (Nx2 ndarray: `[[x0, y0], [x1, y1], ...]`)
        """
        return self._vertices

    @vertices.setter
    def vertices(self, val: numpy.ndarray):
        val = numpy.array(val, dtype=float)                 #TODO document that these might not be copied
        if len(val.shape) < 2 or val.shape[1] != 2:
            raise PatternError('Vertices must be an Nx2 array')
        if val.shape[0] < 2:
            raise PatternError('Must have at least 2 vertices (Nx2 where N>1)')
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
                 width: float = 0.0,
                 *,
                 cap: PathCap = PathCap.Flush,
                 cap_extensions: numpy.ndarray = None,
                 offset: vector2 = (0.0, 0.0),
                 rotation: float = 0,
                 mirrored: Sequence[bool] = (False, False),
                 layer: layer_t = 0,
                 dose: float = 1.0,
                 repetition: Optional[Repetition] = None,
                 annotations: Optional[annotations_t] = None,
                 locked: bool = False,
                 raw: bool = False,
                 ):
        LockableImpl.unlock(self)
        self._cap_extensions = None     # Since .cap setter might access it

        self.identifier = ()
        if raw:
            self._vertices = vertices
            self._offset = offset
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
            self._layer = layer
            self._dose = dose
            self._width = width
            self._cap = cap
            self._cap_extensions = cap_extensions
        else:
            self.vertices = vertices
            self.offset = offset
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
            self.layer = layer
            self.dose = dose
            self.width = width
            self.cap = cap
            self.cap_extensions = cap_extensions
        self.rotate(rotation)
        [self.mirror(a) for a, do in enumerate(mirrored) if do]
        self.set_locked(locked)

    def  __deepcopy__(self, memo: Dict = None) -> 'Path':
        memo = {} if memo is None else memo
        new = copy.copy(self).unlock()
        new._offset = self._offset.copy()
        new._vertices = self._vertices.copy()
        new._cap = copy.deepcopy(self._cap, memo)
        new._cap_extensions = copy.deepcopy(self._cap_extensions, memo)
        new._annotations = copy.deepcopy(self._annotations)
        new.set_locked(self.locked)
        return new

    @staticmethod
    def travel(travel_pairs: Tuple[Tuple[float, float]],
               width: float = 0.0,
               cap: PathCap = PathCap.Flush,
               cap_extensions = None,
               offset: vector2 = (0.0, 0.0),
               rotation: float = 0,
               mirrored: Sequence[bool] = (False, False),
               layer: layer_t = 0,
               dose: float = 1.0,
               ) -> 'Path':
        """
        Build a path by specifying the turn angles and travel distances
          rather than setting the distances directly.

        Args:
            travel_pairs: A list of (angle, distance) pairs that define
                the path. Angles are counterclockwise, in radians, and are relative
                to the previous segment's direction (the initial angle is relative
                to the +x axis).
            width: Path width, default `0`
            cap: End-cap type, default `Path.Cap.Flush` (no end-cap)
            cap_extensions: End-cap extension distances, when using `Path.Cap.CustomSquare`.
                Default `(0, 0)` or `None`, depending on cap type
            offset: Offset, default `(0, 0)`
            rotation: Rotation counterclockwise, in radians. Default `0`
            mirrored: Whether to mirror across the x or y axes. For example,
                `mirrored=(True, False)` results in a reflection across the x-axis,
                multiplying the path's y-coordinates by -1. Default `(False, False)`
            layer: Layer, default `0`
            dose: Dose, default `1.0`

        Returns:
            The resulting Path object
        """
        #TODO: needs testing
        direction = numpy.array([1, 0])

        verts = [[0, 0]]
        for angle, distance in travel_pairs:
            direction = numpy.dot(rotation_matrix_2d(angle), direction.T).T
            verts.append(verts[-1] + direction * distance)

        return Path(vertices=verts, width=width, cap=cap, cap_extensions=cap_extensions,
                    offset=offset, rotation=rotation, mirrored=mirrored,
                    layer=layer, dose=dose)

    def to_polygons(self,
                    poly_num_points: int = None,
                    poly_max_arclen: float = None,
                    ) -> List['Polygon']:
        extensions = self._calculate_cap_extensions()

        v = remove_colinear_vertices(self.vertices, closed_path=False)
        dv = numpy.diff(v, axis=0)
        dvdir = dv / numpy.sqrt((dv * dv).sum(axis=1))[:, None]

        if self.width == 0:
            verts = numpy.vstack((v, v[::-1]))
            return [Polygon(offset=self.offset, vertices=verts, dose=self.dose, layer=self.layer)]

        perp = dvdir[:, ::-1] * [[1, -1]] * self.width / 2

        # add extensions
        if (extensions != 0).any():
            v[0] -= dvdir[0] * extensions[0]
            v[-1] += dvdir[-1] * extensions[1]
            dv = numpy.diff(v, axis=0)      # recalculate dv; dvdir and perp should stay the same

        # Find intersections of expanded sides
        As = numpy.stack((dv[:-1], -dv[1:]), axis=2)
        bs = v[1:-1] - v[:-2] + perp[1:] - perp[:-1]
        ds = v[1:-1] - v[:-2] - perp[1:] + perp[:-1]

        rp = numpy.linalg.solve(As, bs)[:, 0, None]
        rn = numpy.linalg.solve(As, ds)[:, 0, None]

        intersection_p = v[:-2] + rp * dv[:-1] + perp[:-1]
        intersection_n = v[:-2] + rn * dv[:-1] - perp[:-1]

        towards_perp = (dv[1:] * perp[:-1]).sum(axis=1) > 0 # path bends towards previous perp?
#       straight = (dv[1:] * perp[:-1]).sum(axis=1) == 0    # path is straight
        acute = (dv[1:] * dv[:-1]).sum(axis=1) < 0          # angle is acute?

        # Build vertices
        o0 = [v[0] + perp[0]]
        o1 = [v[0] - perp[0]]
        for i in range(dv.shape[0] - 1):
            if towards_perp[i]:
                o0.append(intersection_p[i])
                if acute[i]:
                    # Opposite is >270
                    pt0 = v[i + 1] - perp[i + 0] + dvdir[i + 0] * self.width / 2
                    pt1 = v[i + 1] - perp[i + 1] - dvdir[i + 1] * self.width / 2
                    o1 += [pt0, pt1]
                else:
                    o1.append(intersection_n[i])
            else:
                o1.append(intersection_n[i])
                if acute[i]:
                    # > 270
                    pt0 = v[i + 1] + perp[i + 0] + dvdir[i + 0] * self.width / 2
                    pt1 = v[i + 1] + perp[i + 1] - dvdir[i + 1] * self.width / 2
                    o0 += [pt0, pt1]
                else:
                    o0.append(intersection_p[i])
        o0.append(v[-1] + perp[-1])
        o1.append(v[-1] - perp[-1])
        verts = numpy.vstack((o0, o1[::-1]))

        polys = [Polygon(offset=self.offset, vertices=verts, dose=self.dose, layer=self.layer)]

        if self.cap == PathCap.Circle:
            #for vert in v:         # not sure if every vertex, or just ends?
            for vert in [v[0], v[-1]]:
                circ = Circle(offset=vert, radius=self.width / 2, dose=self.dose, layer=self.layer)
                polys += circ.to_polygons(poly_num_points=poly_num_points, poly_max_arclen=poly_max_arclen)

        return polys

    def get_bounds(self) -> numpy.ndarray:
        if self.cap == PathCap.Circle:
            bounds = self.offset + numpy.vstack((numpy.min(self.vertices, axis=0) - self.width / 2,
                                                 numpy.max(self.vertices, axis=0) + self.width / 2))
        elif self.cap in (PathCap.Flush,
                          PathCap.Square,
                          PathCap.SquareCustom):
            bounds = numpy.array([[+inf, +inf], [-inf, -inf]])
            polys = self.to_polygons()
            for poly in polys:
                poly_bounds = poly.get_bounds()
                bounds[0, :] = numpy.minimum(bounds[0, :], poly_bounds[0, :])
                bounds[1, :] = numpy.maximum(bounds[1, :], poly_bounds[1, :])
        else:
            raise PatternError(f'get_bounds() not implemented for endcaps: {self.cap}')

        return bounds

    def rotate(self, theta: float) -> 'Path':
        if theta != 0:
            self.vertices = numpy.dot(rotation_matrix_2d(theta), self.vertices.T).T
        return self

    def mirror(self, axis: int) -> 'Path':
        self.vertices[:, axis - 1] *= -1
        return self

    def scale_by(self, c: float) -> 'Path':
        self.vertices *= c
        self.width *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        # Note: this function is going to be pretty slow for many-vertexed paths, relative to
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

        width0 = self.width / norm_value

        return (type(self), reordered_vertices.data.tobytes(), width0, self.cap, self.layer), \
               (offset, scale/norm_value, rotation, False, self.dose), \
               lambda: Path(reordered_vertices*norm_value, width=self.width*norm_value,
                            cap=self.cap, layer=self.layer)

    def clean_vertices(self) -> 'Path':
        """
        Removes duplicate, co-linear and otherwise redundant vertices.

        Returns:
            self
        """
        self.remove_colinear_vertices()
        return self

    def remove_duplicate_vertices(self) -> 'Path':
        '''
        Removes all consecutive duplicate (repeated) vertices.

        Returns:
            self
        '''
        self.vertices = remove_duplicate_vertices(self.vertices, closed_path=False)
        return self

    def remove_colinear_vertices(self) -> 'Path':
        '''
        Removes consecutive co-linear vertices.

        Returns:
            self
        '''
        self.vertices = remove_colinear_vertices(self.vertices, closed_path=False)
        return self

    def _calculate_cap_extensions(self) -> numpy.ndarray:
        if self.cap == PathCap.Square:
            extensions = numpy.full(2, self.width / 2)
        elif self.cap == PathCap.SquareCustom:
            extensions =  self.cap_extensions
        else:
            # Flush or Circle
            extensions = numpy.zeros(2)
        return extensions

    def lock(self) -> 'Path':
        self.vertices.flags.writeable = False
        if self.cap_extensions is not None:
            self.cap_extensions.flags.writeable = False
        Shape.lock(self)
        return self

    def unlock(self) -> 'Path':
        Shape.unlock(self)
        self.vertices.flags.writeable = True
        if self.cap_extensions is not None:
            self.cap_extensions.flags.writeable = True
        return self

    def __repr__(self) -> str:
        centroid = self.offset + self.vertices.mean(axis=0)
        dose = f' d{self.dose:g}' if self.dose != 1 else ''
        locked = ' L' if self.locked else ''
        return f'<Path l{self.layer} centroid {centroid} v{len(self.vertices)} w{self.width} c{self.cap}{dose}{locked}>'
