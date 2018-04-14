from typing import List, Tuple
import numpy
from numpy import pi, inf

from . import Shape, Polygon, normalized_shape_tuple
from .. import PatternError
from ..utils import is_scalar, vector2, get_bit

# Loaded on use:
# from freetype import Face
# from matplotlib.path import Path


__author__ = 'Jan Petykiewicz'


class Text(Shape):
    _string = ''
    _height = 1.0
    _rotation = 0.0
    _mirrored = None
    font_path = ''

    # vertices property
    @property
    def string(self) -> str:
        return self._string

    @string.setter
    def string(self, val: str):
        self._string = val

    # Rotation property
    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, val: float):
        if not is_scalar(val):
            raise PatternError('Rotation must be a scalar')
        self._rotation = val % (2 * pi)

    # Height property
    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, val: float):
        if not is_scalar(val):
            raise PatternError('Height must be a scalar')
        self._height = val

    # Mirrored property
    @property
    def mirrored(self) -> List[bool]:
        return self._mirrored

    @mirrored.setter
    def mirrored(self, val: List[bool]):
        if is_scalar(val):
            raise PatternError('Mirrored must be a 2-element list of booleans')
        self._mirrored = val

    def __init__(self,
                 string: str,
                 height: float,
                 font_path: str,
                 mirrored: List[bool]=None,
                 rotation: float=0.0,
                 offset: vector2=(0.0, 0.0),
                 layer: int=0,
                 dose: float=1.0):
        self.offset = offset
        self.layer = layer
        self.dose = dose
        self.string = string
        self.height = height
        self.rotation = rotation
        self.font_path = font_path
        if mirrored is None:
            mirrored = [False, False]
        self.mirrored = mirrored

    def to_polygons(self,
                    _poly_num_points: int=None,
                    _poly_max_arclen: float=None
                    ) -> List[Polygon]:
        all_polygons = []
        total_advance = 0
        for char in self.string:
            raw_polys, advance = get_char_as_polygons(self.font_path, char)

            # Move these polygons to the right of the previous letter
            for xys in raw_polys:
                poly = Polygon(xys, dose=self.dose, layer=self.layer)
                [poly.mirror(ax) for ax, do in enumerate(self.mirrored) if do]
                poly.scale_by(self.height)
                poly.offset = self.offset + [total_advance, 0]
                poly.rotate_around(self.offset, self.rotation)
                all_polygons += [poly]

            # Update the list of all polygons and how far to advance
            total_advance += advance * self.height

        return all_polygons

    def rotate(self, theta: float) -> 'Text':
        self.rotation += theta
        return self

    def mirror(self, axis: int) -> 'Text':
        self.mirrored[axis] = not self.mirrored[axis]
        return self

    def scale_by(self, c: float) -> 'Text':
        self.height *= c
        return self

    def normalized_form(self, norm_value: float) -> normalized_shape_tuple:
        return (type(self), self.string, self.font_path, self.mirrored, self.layer), \
               (self.offset, self.height / norm_value, self.rotation, self.dose), \
               lambda: Text(string=self.string,
                            height=self.height * norm_value,
                            font_path=self.font_path,
                            mirrored=self.mirrored,
                            layer=self.layer)

    def get_bounds(self) -> numpy.ndarray:
        # rotation makes this a huge pain when using slot.advance and glyph.bbox(), so
        #  just convert to polygons instead
        bounds = [[+inf, +inf], [-inf, -inf]]
        polys = self.to_polygons()
        for poly in polys:
            poly_bounds = poly.get_bounds()
            bounds[0, :] = numpy.minimum(bounds[0, :], poly_bounds[0, :])
            bounds[1, :] = numpy.maximum(bounds[1, :], poly_bounds[1, :])

        return bounds


def get_char_as_polygons(font_path: str,
                         char: str,
                         resolution: float=48*64,
                         ) -> Tuple[List[List[List[float]]], float]:
    from freetype import Face
    from matplotlib.path import Path

    """
    Get a list of polygons representing a single character.

    The output is normalized so that the font size is 1 unit.

    :param font_path: File path specifying a font loadable by freetype
    :param char: Character to convert to polygons
    :param resolution: Internal resolution setting (used for freetype
            Face.set_font_size(resolution)). Modify at your own peril!
    :return: List of polygons [[[x0, y0], [x1, y1], ...], ...] and 'advance' distance (distance
            from the start of this glyph to the start of the next one)
    """
    if len(char) != 1:
        raise Exception('get_char_as_polygons called with non-char')

    face = Face(font_path)
    face.set_char_size(resolution)
    face.load_char(char)
    slot = face.glyph
    outline = slot.outline

    start = 0
    all_verts, all_codes = [], []
    for end in outline.contours:
        points = outline.points[start:end + 1]
        points.append(points[0])

        tags = outline.tags[start:end + 1]
        tags.append(tags[0])

        segments = []
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
        all_verts.extend(verts)
        all_codes.extend(codes)
        start = end + 1

    all_verts = numpy.array(all_verts) / resolution

    advance = slot.advance.x / resolution

    if len(all_verts) == 0:
        polygons = []
    else:
        path = Path(all_verts, all_codes)
        path.should_simplify = False
        polygons = path.to_polygons()

    return polygons, advance
