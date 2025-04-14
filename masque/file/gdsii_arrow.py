"""
GDSII file format readers and writers using the `klamath` library.

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
import pyarrow
from pyarrow.cffi import ffi

from .utils import is_gzipped, tmpfile
from .. import Pattern, Ref, PatternError, LibraryError, Label, Shape
from ..shapes import Polygon, Path
from ..repetition import Grid
from ..utils import layer_t, annotations_t
from ..library import LazyLibrary, Library, ILibrary, ILibraryView


logger = logging.getLogger(__name__)

clib = ffi.dlopen('/home/jan/projects/klamath-rs/target/debug/libklamath_rs_ext.so')
ffi.cdef('void read_path(char* path, struct ArrowArray* array, struct ArrowSchema* schema);')


path_cap_map = {
    0: Path.Cap.Flush,
    1: Path.Cap.Circle,
    2: Path.Cap.Square,
    4: Path.Cap.SquareCustom,
    }


def rint_cast(val: ArrayLike) -> NDArray[numpy.int32]:
    return numpy.rint(val).astype(numpy.int32)


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
    path = pathlib.Path(filename)
    path.resolve()
    ptr_array = ffi.new('struct ArrowArray[]', 1)
    ptr_schema = ffi.new('struct ArrowSchema[]', 1)
    clib.read_path(str(path).encode(), ptr_array, ptr_schema)

    iptr_schema = int(ffi.cast('uintptr_t', ptr_schema))
    iptr_array = int(ffi.cast('uintptr_t', ptr_array))
    arrow_arr = pyarrow.Array._import_from_c(iptr_array, iptr_schema)
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

    mlib = Library()
    for cell in libarr['cells']:
        name = libarr['cell_names'][cell['id'].as_py()].as_py()
        pat = read_cell(cell, libarr['cell_names'], raw_mode=raw_mode)
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


def read_cell(
        cellarr: pyarrow.Array,
        cell_names: pyarrow.Array,
        raw_mode: bool = True,
        ) -> Pattern:
    """
    TODO
    Read elements from a GDS structure and build a Pattern from them.

    Args:
        stream: Seekable stream, positioned at a record boundary.
                Will be read until an ENDSTR record is consumed.
        name: Name of the resulting Pattern
        raw_mode: If True, bypass per-shape data validation. Default True.

    Returns:
        A pattern containing the elements that were read.
    """
    pat = Pattern()

    for refarr in cellarr['refs']:
        target = cell_names[refarr['target'].as_py()].as_py()
        args = dict(
            offset = (refarr['x'].as_py(), refarr['y'].as_py()),
            )
        if (mirr := refarr['invert_y']).is_valid:
            args['mirrored'] = mirr.as_py()
        if (rot := refarr['angle_deg']).is_valid:
            args['rotation'] = numpy.deg2rad(rot.as_py())
        if (mag := refarr['mag']).is_valid:
            args['scale'] = mag.as_py()
        if (rep := refarr['repetition']).is_valid:
            repetition = Grid(
                a_vector = (rep['x0'].as_py(), rep['y0'].as_py()),
                b_vector = (rep['x1'].as_py(), rep['y1'].as_py()),
                a_count = rep['count0'].as_py(),
                b_count = rep['count1'].as_py(),
                )
            args['repetition'] = repetition
        ref = Ref(**args)
        pat.refs[target].append(ref)

    for bnd in cellarr['boundaries']:
        layer = (bnd['layer'].as_py(), bnd['dtype'].as_py())
        args = dict(
            vertices = bnd['xy'].values.to_numpy().reshape((-1, 2))[:-1],
            )

        if (props := bnd['properties']).is_valid:
            args['annotations'] = _properties_to_annotations(props)

        poly = Polygon(**args)
        pat.shapes[layer].append(poly)

    for gpath in cellarr['paths']:
        layer = (gpath['layer'].as_py(), gpath['dtype'].as_py())
        args = dict(
            vertices = gpath['xy'].values.to_numpy().reshape((-1, 2)),
            )

        if (gcap := gpath['path_type']).is_valid:
            mcap = path_cap_map[gcap.as_py()]
            args['cap'] = mcap
            if mcap == Path.Cap.SquareCustom:
                extensions = [0, 0]
                if (ext0 := gpath['extension_start']).is_valid:
                    extensions[0] = ext0.as_py()
                if (ext1 := gpath['extension_end']).is_valid:
                    extensions[1] = ext1.as_py()

                args['extensions'] = extensions

        if (width := gpath['width']).is_valid:
            args['width'] = width.as_py()
        else:
            args['width'] = 0

        if (props := gpath['properties']).is_valid:
            args['annotations'] = _properties_to_annotations(props)

        mpath = Path(**args)
        pat.shapes[layer].append(mpath)

    for gtext in cellarr['texts']:
        layer = (gtext['layer'].as_py(), gtext['dtype'].as_py())
        args = dict(
            offset = (gtext['x'].as_py(), gtext['y'].as_py()),
            string = gtext['string'].as_py(),
            )

        if (props := gtext['properties']).is_valid:
            args['annotations'] = _properties_to_annotations(props)

        mlabel = Label(**args)
        pat.labels[layer].append(mlabel)

    return pat


def _properties_to_annotations(properties: pyarrow.Array) -> annotations_t:
    return {prop['key'].as_py(): prop['value'].as_py() for prop in properties}


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
