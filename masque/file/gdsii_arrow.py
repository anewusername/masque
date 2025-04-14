# ruff: noqa: ARG001, F401
"""
GDSII file format readers and writers using the `TODO` library.

Note that GDSII references follow the same convention as `masque`,
  with this order of operations:
   1. Mirroring
   2. Rotation
   3. Scaling
   4. Offset and array expansion (no mirroring/rotation/scaling applied to offsets)

  Scaling, rotation, and mirroring apply to individual instances, not grid
   vectors or offsets.

Notes:
 * absolute positioning is not supported
 * PLEX is not supported
 * ELFLAGS are not supported
 * GDS does not support library- or structure-level annotations
 * GDS creation/modification/access times are set to 1900-01-01 for reproducibility.
 * Gzip modification time is set to 0 (start of current epoch, usually 1970-01-01)

 TODO writing
 TODO warn on boxes, nodes
"""
from typing import IO, cast, Any
from collections.abc import Iterable, Mapping, Callable
import io
import mmap
import logging
import pathlib
import gzip
import string
from pprint import pformat

import numpy
from numpy.typing import ArrayLike, NDArray
from numpy.testing import assert_equal
import pyarrow
from pyarrow.cffi import ffi

from .utils import is_gzipped, tmpfile
from .. import Pattern, Ref, PatternError, LibraryError, Label, Shape
from ..shapes import Polygon, Path, PolyCollection
from ..repetition import Grid
from ..utils import layer_t, annotations_t
from ..library import LazyLibrary, Library, ILibrary, ILibraryView


logger = logging.getLogger(__name__)

clib = ffi.dlopen('/home/jan/projects/klamath-rs/target/release/libklamath_rs_ext.so')
ffi.cdef('void read_path(char* path, struct ArrowArray* array, struct ArrowSchema* schema);')


path_cap_map = {
    0: Path.Cap.Flush,
    1: Path.Cap.Circle,
    2: Path.Cap.Square,
    4: Path.Cap.SquareCustom,
    }


def rint_cast(val: ArrayLike) -> NDArray[numpy.int32]:
    return numpy.rint(val).astype(numpy.int32)


def _read_to_arrow(
        filename: str | pathlib.Path,
        *args,
        **kwargs,
        ) -> pyarrow.Array:
    path = pathlib.Path(filename)
    path.resolve()
    ptr_array = ffi.new('struct ArrowArray[]', 1)
    ptr_schema = ffi.new('struct ArrowSchema[]', 1)
    clib.read_path(str(path).encode(), ptr_array, ptr_schema)

    iptr_schema = int(ffi.cast('uintptr_t', ptr_schema))
    iptr_array = int(ffi.cast('uintptr_t', ptr_array))
    arrow_arr = pyarrow.Array._import_from_c(iptr_array, iptr_schema)

    return arrow_arr


def readfile(
        filename: str | pathlib.Path,
        *args,
        **kwargs,
        ) -> tuple[Library, dict[str, Any]]:
    """
    Wrapper for `read()` that takes a filename or path instead of a stream.

    Will automatically decompress gzipped files.

    Args:
        filename: Filename to save to.
        *args: passed to `read()`
        **kwargs: passed to `read()`
    """
    arrow_arr = _read_to_arrow(filename)
    assert len(arrow_arr) == 1

    results = read_arrow(arrow_arr[0])

    return results


def read_arrow(
        libarr: pyarrow.Array,
        raw_mode: bool = True,
        ) -> tuple[Library, dict[str, Any]]:
    """
    # TODO check GDSII file for cycles!
    Read a gdsii file and translate it into a dict of Pattern objects. GDSII structures are
     translated into Pattern objects; boundaries are translated into polygons, and srefs and arefs
     are translated into Ref objects.

    Additional library info is returned in a dict, containing:
      'name': name of the library
      'meters_per_unit': number of meters per database unit (all values are in database units)
      'logical_units_per_unit': number of "logical" units displayed by layout tools (typically microns)
                                per database unit

    Args:
        stream: Stream to read from.
        raw_mode: If True, constructs shapes in raw mode, bypassing most data validation, Default True.

    Returns:
        - dict of pattern_name:Patterns generated from GDSII structures
        - dict of GDSII library info
    """
    library_info = _read_header(libarr)

    layer_names_np = libarr['layers'].values.to_numpy().view('i2').reshape((-1, 2))
    layer_tups = [tuple(pair) for pair in layer_names_np]

    cell_ids = libarr['cells'].values.field('id').to_numpy()
    cell_names = libarr['cell_names'].as_py()

    def get_geom(libarr: pyarrow.Array, geom_type: str) -> dict[str, Any]:
        el = libarr['cells'].values.field(geom_type)
        elem = dict(
            offsets = el.offsets.to_numpy(),
            xy_arr = el.values.field('xy').values.to_numpy().reshape((-1, 2)),
            xy_off = el.values.field('xy').offsets.to_numpy() // 2,
            layer_inds = el.values.field('layer').to_numpy(),
            prop_off = el.values.field('properties').offsets.to_numpy(),
            prop_key = el.values.field('properties').values.field('key').to_numpy(),
            prop_val = el.values.field('properties').values.field('value').to_pylist(),
            )
        return elem

    rf = libarr['cells'].values.field('refs')
    refs = dict(
        offsets = rf.offsets.to_numpy(),
        targets = rf.values.field('target').to_numpy(),
        xy = rf.values.field('xy').to_numpy().view('i4').reshape((-1, 2)),
        invert_y = rf.values.field('invert_y').fill_null(False).to_numpy(zero_copy_only=False),
        angle_rad = numpy.rad2deg(rf.values.field('angle_deg').fill_null(0).to_numpy()),
        scale = rf.values.field('mag').fill_null(1).to_numpy(),
        rep_valid = rf.values.field('repetition').is_valid().to_numpy(zero_copy_only=False),
        rep_xy0 = rf.values.field('repetition').field('xy0').fill_null(0).to_numpy().view('i4').reshape((-1, 2)),
        rep_xy1 = rf.values.field('repetition').field('xy1').fill_null(0).to_numpy().view('i4').reshape((-1, 2)),
        rep_counts = rf.values.field('repetition').field('counts').fill_null(0).to_numpy().view('i2').reshape((-1, 2)),
        prop_off = rf.values.field('properties').offsets.to_numpy(),
        prop_key = rf.values.field('properties').values.field('key').to_numpy(),
        prop_val = rf.values.field('properties').values.field('value').to_pylist(),
        )

    txt = libarr['cells'].values.field('texts')
    texts = dict(
        offsets = txt.offsets.to_numpy(),
        layer_inds = txt.values.field('layer').to_numpy(),
        xy = txt.values.field('xy').to_numpy().view('i4').reshape((-1, 2)),
        string = txt.values.field('string').to_pylist(),
        prop_off = txt.values.field('properties').offsets.to_numpy(),
        prop_key = txt.values.field('properties').values.field('key').to_numpy(),
        prop_val = txt.values.field('properties').values.field('value').to_pylist(),
        )

    elements = dict(
        boundaries = get_geom(libarr, 'boundaries'),
        paths = get_geom(libarr, 'paths'),
        boxes = get_geom(libarr, 'boxes'),
        nodes = get_geom(libarr, 'nodes'),
        texts = texts,
        refs = refs,
        )

    paths = libarr['cells'].values.field('paths')
    elements['paths'].update(dict(
        width = paths.values.field('width').fill_null(0).to_numpy(),
        path_type = paths.values.field('path_type').fill_null(0).to_numpy(),
        extensions = numpy.stack((
            paths.values.field('extension_start').fill_null(0).to_numpy(),
            paths.values.field('extension_end').fill_null(0).to_numpy(),
            ), axis=-1),
        ))

    global_args = dict(
        cell_names = cell_names,
        layer_tups = layer_tups,
        raw_mode = raw_mode,
        )

    mlib = Library()
    for cc in range(len(libarr['cells'])):
        name = cell_names[cell_ids[cc]]
        pat = Pattern()
        _boundaries_to_polygons(pat, global_args, elements['boundaries'], cc)
        _gpaths_to_mpaths(pat, global_args, elements['paths'], cc)
        _grefs_to_mrefs(pat, global_args, elements['refs'], cc)
        _texts_to_labels(pat, global_args, elements['texts'], cc)
        mlib[name] = pat

    return mlib, library_info


def _read_header(libarr: pyarrow.Array) -> dict[str, Any]:
    """
    Read the file header and create the library_info dict.
    """
    library_info = dict(
        name = libarr['lib_name'],
        meters_per_unit = libarr['meters_per_db_unit'],
        logical_units_per_unit = libarr['user_units_per_db_unit'],
        )
    return library_info


def _grefs_to_mrefs(
        pat: Pattern,
        global_args: dict[str, Any],
        elem: dict[str, Any],
        cc: int,
        ) -> None:
    cell_names = global_args['cell_names']
    elem_off = elem['offsets']      # which elements belong to each cell
    xy = elem['xy']
    prop_key = elem['prop_key']
    prop_val = elem['prop_val']
    targets = elem['targets']

    elem_count = elem_off[cc + 1] - elem_off[cc]
    elem_slc = slice(elem_off[cc], elem_off[cc] + elem_count + 1)   # +1 to capture ending location for last elem
    prop_offs = elem['prop_off'][elem_slc]  # which props belong to each element
    elem_invert_y = elem['invert_y'][elem_slc][:elem_count]
    elem_angle_rad = elem['angle_rad'][elem_slc][:elem_count]
    elem_scale = elem['scale'][elem_slc][:elem_count]
    elem_rep_xy0 = elem['rep_xy0'][elem_slc][:elem_count]
    elem_rep_xy1 = elem['rep_xy1'][elem_slc][:elem_count]
    elem_rep_counts = elem['rep_counts'][elem_slc][:elem_count]
    rep_valid = elem['rep_valid'][elem_slc][:elem_count]


    for ee in range(elem_count):
        target = cell_names[targets[ee]]
        offset = xy[ee]
        mirr = elem_invert_y[ee]
        rot = elem_angle_rad[ee]
        mag = elem_scale[ee]

        rep: None | Grid = None
        if rep_valid[ee]:
            a_vector = elem_rep_xy0[ee]
            b_vector = elem_rep_xy1[ee]
            a_count, b_count = elem_rep_counts[ee]
            rep = Grid(a_vector=a_vector, b_vector=b_vector, a_count=a_count, b_count=b_count)

        annotations: None | dict[str, list[int | float | str]] = None
        prop_ii, prop_ff = prop_offs[ee], prop_offs[ee + 1]
        if prop_ii < prop_ff:
            annotations = {str(prop_key[off]): [prop_val[off]] for off in range(prop_ii, prop_ff)}

        ref = Ref(offset=offset, mirrored=mirr, rotation=rot, scale=mag, repetition=rep, annotations=annotations)
        pat.refs[target].append(ref)


def _texts_to_labels(
        pat: Pattern,
        global_args: dict[str, Any],
        elem: dict[str, Any],
        cc: int,
        ) -> None:
    elem_off = elem['offsets']      # which elements belong to each cell
    xy = elem['xy']
    layer_tups = global_args['layer_tups']
    layer_inds = elem['layer_inds']
    prop_key = elem['prop_key']
    prop_val = elem['prop_val']

    elem_count = elem_off[cc + 1] - elem_off[cc]
    elem_slc = slice(elem_off[cc], elem_off[cc] + elem_count + 1)   # +1 to capture ending location for last elem
    prop_offs = elem['prop_off'][elem_slc]  # which props belong to each element
    elem_layer_inds = layer_inds[elem_slc][:elem_count]
    elem_strings = elem['string'][elem_slc][:elem_count]

    for ee in range(elem_count):
        layer = layer_tups[elem_layer_inds[ee]]
        offset = xy[ee]
        string = elem_strings[ee]

        annotations: None | dict[str, list[int | float | str]] = None
        prop_ii, prop_ff = prop_offs[ee], prop_offs[ee + 1]
        if prop_ii < prop_ff:
            annotations = {str(prop_key[off]): [prop_val[off]] for off in range(prop_ii, prop_ff)}

        mlabel = Label(string=string, offset=offset, annotations=annotations)
        pat.labels[layer].append(mlabel)


def _gpaths_to_mpaths(
        pat: Pattern,
        global_args: dict[str, Any],
        elem: dict[str, Any],
        cc: int,
        ) -> None:
    elem_off = elem['offsets']      # which elements belong to each cell
    xy_val = elem['xy_arr']
    layer_tups = global_args['layer_tups']
    layer_inds = elem['layer_inds']
    prop_key = elem['prop_key']
    prop_val = elem['prop_val']

    elem_count = elem_off[cc + 1] - elem_off[cc]
    elem_slc = slice(elem_off[cc], elem_off[cc] + elem_count + 1)   # +1 to capture ending location for last elem
    xy_offs = elem['xy_off'][elem_slc]      # which xy coords belong to each element
    prop_offs = elem['prop_off'][elem_slc]  # which props belong to each element
    elem_layer_inds = layer_inds[elem_slc][:elem_count]
    elem_widths = elem['width'][elem_slc][:elem_count]
    elem_path_types = elem['path_type'][elem_slc][:elem_count]
    elem_extensions = elem['extensions'][elem_slc][:elem_count]

    zeros = numpy.zeros((elem_count, 2))
    raw_mode = global_args['raw_mode']
    for ee in range(elem_count):
        layer = layer_tups[elem_layer_inds[ee]]
        vertices = xy_val[xy_offs[ee]:xy_offs[ee + 1]]
        width = elem_widths[ee]
        cap_int = elem_path_types[ee]
        cap = path_cap_map[cap_int]
        if cap_int == 4:
            cap_extensions = elem_extensions[ee]
        else:
            cap_extensions = None

        annotations: None | dict[str, list[int | float | str]] = None
        prop_ii, prop_ff = prop_offs[ee], prop_offs[ee + 1]
        if prop_ii < prop_ff:
            annotations = {str(prop_key[off]): [prop_val[off]] for off in range(prop_ii, prop_ff)}

        path = Path(vertices=vertices, offset=zeros[ee], annotations=annotations, raw=raw_mode,
            width=width, cap=cap,cap_extensions=cap_extensions)
        pat.shapes[layer].append(path)


def _boundaries_to_polygons(
        pat: Pattern,
        global_args: dict[str, Any],
        elem: dict[str, Any],
        cc: int,
        ) -> None:
    elem_off = elem['offsets']      # which elements belong to each cell
    xy_val = elem['xy_arr']
    layer_inds = elem['layer_inds']
    layer_tups = global_args['layer_tups']
    prop_key = elem['prop_key']
    prop_val = elem['prop_val']

    elem_count = elem_off[cc + 1] - elem_off[cc]
    elem_slc = slice(elem_off[cc], elem_off[cc] + elem_count + 1)   # +1 to capture ending location for last elem
    xy_offs = elem['xy_off'][elem_slc]      # which xy coords belong to each element
    xy_counts = xy_offs[1:] - xy_offs[:-1]
    prop_offs = elem['prop_off'][elem_slc]  # which props belong to each element
    prop_counts = prop_offs[1:] - prop_offs[:-1]
    elem_layer_inds = layer_inds[elem_slc][:elem_count]

    order = numpy.argsort(elem_layer_inds, stable=True)
    unilayer_inds, unilayer_first, unilayer_count = numpy.unique(elem_layer_inds, return_index=True, return_counts=True)

    zeros = numpy.zeros((elem_count, 2))
    raw_mode = global_args['raw_mode']
    for layer_ind, ff, nn in zip(unilayer_inds, unilayer_first, unilayer_count, strict=True):
        ee_inds = order[ff:ff + nn]
        layer = layer_tups[layer_ind]
        propless_mask = prop_counts[ee_inds] == 0

        poly_count_on_layer = propless_mask.sum()
        if poly_count_on_layer == 1:
            propless_mask[:] = 0        # Never make a 1-element collection
        elif poly_count_on_layer > 1:
            propless_vert_counts = xy_counts[ee_inds[propless_mask]] - 1        # -1 to drop closing point
            vertex_lists = numpy.empty((propless_vert_counts.sum(), 2), dtype=numpy.float64)
            vertex_offsets = numpy.cumsum(numpy.concatenate([[0], propless_vert_counts]))

            for ii, ee in enumerate(ee_inds[propless_mask]):
                vo = vertex_offsets[ii]
                vertex_lists[vo:vo + propless_vert_counts[ii]] = xy_val[xy_offs[ee]:xy_offs[ee + 1] - 1]

            polys = PolyCollection(vertex_lists=vertex_lists, vertex_offsets=vertex_offsets, offset=zeros[ee])
            pat.shapes[layer].append(polys)

        # Handle single polygons
        for ee in ee_inds[~propless_mask]:
            layer = layer_tups[elem_layer_inds[ee]]
            vertices = xy_val[xy_offs[ee]:xy_offs[ee + 1] - 1]    # -1 to drop closing point

            annotations: None | dict[str, list[int | float | str]] = None
            prop_ii, prop_ff = prop_offs[ee], prop_offs[ee + 1]
            if prop_ii < prop_ff:
                annotations = {str(prop_key[off]): prop_val[off] for off in range(prop_ii, prop_ff)}

            poly = Polygon(vertices=vertices, offset=zeros[ee], annotations=annotations, raw=raw_mode)
            pat.shapes[layer].append(poly)


#def _properties_to_annotations(properties: pyarrow.Array) -> annotations_t:
#    return {prop['key'].as_py(): prop['value'].as_py() for prop in properties}


def check_valid_names(
        names: Iterable[str],
        max_length: int = 32,
        ) -> None:
    """
    Check all provided names to see if they're valid GDSII cell names.

    Args:
        names: Collection of names to check
        max_length: Max allowed length

    """
    allowed_chars = set(string.ascii_letters + string.digits + '_?$')

    bad_chars = [
        name for name in names
        if not set(name).issubset(allowed_chars)
        ]

    bad_lengths = [
        name for name in names
        if len(name) > max_length
        ]

    if bad_chars:
        logger.error('Names contain invalid characters:\n' + pformat(bad_chars))

    if bad_lengths:
        logger.error(f'Names too long (>{max_length}:\n' + pformat(bad_chars))

    if bad_chars or bad_lengths:
        raise LibraryError('Library contains invalid names, see log above')
