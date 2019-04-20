"""
Shapes for use with the Pattern class, as well as the Shape abstract class from
 which they are derived.
"""

from .shape import Shape, normalized_shape_tuple, DEFAULT_POLY_NUM_POINTS

from .polygon import Polygon
from .circle import Circle
from .ellipse import Ellipse
from .arc import Arc
from .text import Text
from .path import Path
