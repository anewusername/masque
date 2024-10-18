"""
Shapes for use with the Pattern class, as well as the Shape abstract class from
 which they are derived.
"""

from .shape import (
    Shape as Shape,
    normalized_shape_tuple as normalized_shape_tuple,
    DEFAULT_POLY_NUM_VERTICES as DEFAULT_POLY_NUM_VERTICES,
    )

from .polygon import Polygon as Polygon
from .poly_collection import PolyCollection as PolyCollection
from .circle import Circle as Circle
from .ellipse import Ellipse as Ellipse
from .arc import Arc as Arc
from .text import Text as Text
from .path import Path as Path
