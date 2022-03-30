"""
Various helper functions, type definitions, etc.
"""
from .types import layer_t, annotations_t

from .array import is_scalar
from .autoslots import AutoSlots

from .bitwise import get_bit, set_bit
from .vertices import (
    remove_duplicate_vertices, remove_colinear_vertices, poly_contains_points
    )
from .transform import rotation_matrix_2d, normalize_mirror

#from . import pack2d
