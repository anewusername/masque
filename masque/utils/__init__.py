"""
Various helper functions, type definitions, etc.
"""
from .types import layer_t, annotations_t, SupportsBool
from .array import is_scalar
from .autoslots import AutoSlots
from .deferreddict import DeferredDict
from .decorators import oneshot

from .bitwise import get_bit, set_bit
from .vertices import (
    remove_duplicate_vertices, remove_colinear_vertices, poly_contains_points
    )
from .transform import rotation_matrix_2d, normalize_mirror, rotate_offsets_around

from . import ports2data

from . import pack2d
