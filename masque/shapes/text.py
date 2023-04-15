from typing import Self
import copy

import numpy
from numpy import pi, nan
from numpy.typing import NDArray, ArrayLike

from . import Shape, Polygon, normalized_shape_tuple
from ..error import PatternError
from ..repetition import Repetition
from ..traits import RotatableImpl
from ..utils import is_scalar, get_bit, annotations_t

# Loaded on use:
# from freetype import Face
# from matplotlib.path import Path


class Text(RotatableImpl, Shape):
    """
    Text (to be printed e.g. as a set of polygons).
    This is distinct from non-printed Label objects.
    """
    __slots__ = (
        '_string', '_height', '_mirrored', 'font_path',
        # Inherited
        '_offset', '_repetition', '_annotations', '_rotation',
        )

    _string: str
    _height: float
    _mirrored: bool
    font_path: str

    # vertices property
    @property
    def string(self) -> str:
        return self._string

    @string.setter
    def string(self, val: str) -> None:
        self._string = val

    # Height property
    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, val: float) -> None:
        if not is_scalar(val):
            raise PatternError('Height must be a scalar')
        self._height = val

    @property
    def mirrored(self) -> bool:     # mypy#3004, should be bool
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: bool) -> None:
        self._mirrored = bool(val)

    def __init__(
            self,
            string: str,
            height: float,
            font_path: str,
            *,
            offset: ArrayLike = (0.0, 0.0),
            rotation: float = 0.0,
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            raw: bool = False,
            ) -> None:
        if raw:
            assert isinstance(offset, numpy.ndarray)
            self._offset = offset
            self._string = string
            self._height = height
            self._rotation = rotation
            self._repetition = repetition
            self._annotations = annotations if annotations is not None else {}
        else:
            self.offset = offset
            self.string = string
            self.height = height
            self.rotation = rotation
            self.repetition = repetition
            self.annotations = annotations if annotations is not None else {}
        self.font_path = font_path

    def __deepcopy__(self, memo: dict | None = None) -> Self:
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        new._annotations = copy.deepcopy(self._annotations)
        return new

    def to_polygons(
            self,
            num_vertices: int | None = None,      # unused
            max_arclen: float | None = None,      # unused
            ) -> list[Polygon]:
        all_polygons = []
        total_advance = 0.0
        for char in self.string:
            raw_polys, advance = get_char_as_polygons(self.font_path, char)

            # Move these polygons to the right of the previous letter
            for xys in raw_polys:
                poly = Polygon(xys)
                if self.mirrored:
                    poly.mirror()
                poly.scale_by(self.height)
                poly.offset = self.offset + [total_advance, 0]
                poly.rotate_around(self.offset, self.rotation)
                all_polygons += [poly]

            # Update the list of all polygons and how far to advance
            total_advance += advance * self.height

        return all_polygons

    def mirror(self, axis: int = 0) -> Self:
        self.mirrored = not self.mirrored
        if axis == 1:
            self.rotation += pi
        return self

    def scale_by(self, c: float) -> Self:
        self.height *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        rotation = self.rotation % (2 * pi)
        return ((type(self), self.string, self.font_path),
                (self.offset, self.height / norm_value, rotation, bool(self.mirrored)),
                lambda: Text(
                    string=self.string,
                    height=self.height * norm_value,
                    font_path=self.font_path,
                    rotation=rotation,
                    ).mirror2d(across_x=self.mirrored),
                )

    def get_bounds_single(self) -> NDArray[numpy.float64]:
        # rotation makes this a huge pain when using slot.advance and glyph.bbox(), so
        #  just convert to polygons instead
        polys = self.to_polygons()
        pbounds = numpy.full((len(polys), 2, 2), nan)
        for pp, poly in enumerate(polys):
            pbounds[pp] = poly.get_bounds_nonempty()
        bounds = numpy.vstack((
            numpy.min(pbounds[: 0, :], axis=0),
            numpy.max(pbounds[: 1, :], axis=0),
            ))

        return bounds


def get_char_as_polygons(
        font_path: str,
        char: str,
        resolution: float = 48 * 64,
        ) -> tuple[list[list[list[float]]], float]:
    from freetype import Face           # type: ignore
    from matplotlib.path import Path    # type: ignore

    """
    Get a list of polygons representing a single character.

    The output is normalized so that the font size is 1 unit.

    Args:
        font_path: File path specifying a font loadable by freetype
        char: Character to convert to polygons
        resolution: Internal resolution setting (used for freetype
            `Face.set_font_size(resolution))`. Modify at your own peril!

    Returns:
        List of polygons `[[[x0, y0], [x1, y1], ...], ...]` and
        'advance' distance (distance from the start of this glyph to the start of the next one)
    """
    if len(char) != 1:
        raise Exception('get_char_as_polygons called with non-char')

    face = Face(font_path)
    face.set_char_size(resolution)
    face.load_char(char)
    slot = face.glyph
    outline = slot.outline

    start = 0
    all_verts_list, all_codes = [], []
    for end in outline.contours:
        points = outline.points[start:end + 1]
        points.append(points[0])

        tags = outline.tags[start:end + 1]
        tags.append(tags[0])

        segments: list[list[list[float]]] = []
        for j, point in enumerate(points):
            # If we already have a segment, add this point to it
            if j > 0:
                segments[-1].append(point)

            # If not bezier control point, start next segment
            if get_bit(tags[j], 0) and j < (len(points) - 1):
                segments.append([point])

        verts = [points[0]]
        codes = [Path.MOVETO]
        for segment in segments:
            if len(segment) == 2:
                verts.extend(segment[1:])
                codes.extend([Path.LINETO])
            elif len(segment) == 3:
                verts.extend(segment[1:])
                codes.extend([Path.CURVE3, Path.CURVE3])
            else:
                verts.append(segment[1])
                codes.append(Path.CURVE3)
                for i in range(1, len(segment) - 2):
                    a, b = segment[i], segment[i + 1]
                    c = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                    verts.extend([c, b])
                    codes.extend([Path.CURVE3, Path.CURVE3])
                verts.append(segment[-1])
                codes.append(Path.CURVE3)
        all_verts_list.extend(verts)
        all_codes.extend(codes)
        start = end + 1

    all_verts = numpy.array(all_verts_list) / resolution

    advance = slot.advance.x / resolution

    if len(all_verts) == 0:
        polygons = []
    else:
        path = Path(all_verts, all_codes)
        path.should_simplify = False
        polygons = path.to_polygons()

    return polygons, advance

    def __repr__(self) -> str:
        rotation = f' r°{numpy.rad2deg(self.rotation):g}' if self.rotation != 0 else ''
        mirrored = ' m{:d}' if self.mirrored else ''
        return f'<TextShape "{self.string}" o{self.offset} h{self.height:g}{rotation}{mirrored}>'
