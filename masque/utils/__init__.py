"""
Various helper functions, type definitions, etc.
"""
from .types import (
    layer_t as layer_t,
    annotations_t as annotations_t,
    SupportsBool as SupportsBool,
    )
from .array import is_scalar as is_scalar
from .autoslots import AutoSlots as AutoSlots
from .deferreddict import DeferredDict as DeferredDict
from .decorators import oneshot as oneshot

from .bitwise import (
    get_bit as get_bit,
    set_bit as set_bit,
    )
from .vertices import (
    remove_duplicate_vertices as remove_duplicate_vertices,
    remove_colinear_vertices as remove_colinear_vertices,
    poly_contains_points as poly_contains_points,
    )
from .transform import (
    rotation_matrix_2d as rotation_matrix_2d,
    normalize_mirror as normalize_mirror,
    rotate_offsets_around as rotate_offsets_around,
    )
from .comparisons import (
    annotation2key as annotation2key,
    annotations_lt as annotations_lt,
    annotations_eq as annotations_eq,
    layer2key as layer2key,
    ports_lt as ports_lt,
    ports_eq as ports_eq,
    rep2key as rep2key,
    )

from . import ports2data as ports2data

from . import pack2d as pack2d
