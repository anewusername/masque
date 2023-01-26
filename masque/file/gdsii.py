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
from typing import List, Dict, Tuple, Callable, Union, Iterable, Mapping
from typing import BinaryIO, cast, Optional, Any
import io
import mmap
import logging
import pathlib
import gzip
import string
from pprint import pformat

import numpy
from numpy.typing import ArrayLike, NDArray
import klamath
from klamath import records

from .utils import is_gzipped
from .. import Pattern, Ref, PatternError, LibraryError, Label, Shape
from ..shapes import Polygon, Path
from ..repetition import Grid
from ..utils import layer_t, normalize_mirror, annotations_t
from ..library import LazyLibrary, WrapLibrary, MutableLibrary, Library


logger = logging.getLogger(__name__)


path_cap_map = {
    0: Path.Cap.Flush,
    1: Path.Cap.Circle,
    2: Path.Cap.Square,
    4: Path.Cap.SquareCustom,
    }


def rint_cast(val: ArrayLike) -> NDArray[numpy.int32]:
    return numpy.rint(val).astype(numpy.int32)


def write(
        library: Mapping[str, Pattern],
        stream: BinaryIO,
        meters_per_unit: float,
        logical_units_per_unit: float = 1,
        library_name: str = 'masque-klamath',
        ) -> None:
    """
    Convert a library to a GDSII stream, mapping data as follows:
         Pattern -> GDSII structure
         Ref -> GDSII SREF or AREF
         Path -> GSDII path
         Shape (other than path) -> GDSII boundary/ies
         Label -> GDSII text
         annnotations -> properties, where possible

     For each shape,
        layer is chosen to be equal to `shape.layer` if it is an int,
            or `shape.layer[0]` if it is a tuple
        datatype is chosen to be `shape.layer[1]` if available,
            otherwise `0`

    GDS does not support shape repetition (only cell repeptition). Please call
    `library.wrap_repeated_shapes()` before writing to file.

    Other functions you may want to call:
        - `masque.file.gdsii.check_valid_names(library.keys())` to check for invalid names
        - `library.dangling_refs()` to check for references to missing patterns
        - `pattern.polygonize()` for any patterns with shapes other
            than `masque.shapes.Polygon` or `masque.shapes.Path`

    Args:
        library: A {name: Pattern} mapping of patterns to write.
        meters_per_unit: Written into the GDSII file, meters per (database) length unit.
            All distances are assumed to be an integer multiple of this unit, and are stored as such.
        logical_units_per_unit: Written into the GDSII file. Allows the GDSII to specify a
            "logical" unit which is different from the "database" unit, for display purposes.
            Default `1`.
        library_name: Library name written into the GDSII file.
            Default 'masque-klamath'.
    """
    if not isinstance(library, MutableLibrary):
        if isinstance(library, dict):
            library = WrapLibrary(library)
        else:
            library = WrapLibrary(dict(library))

    # Create library
    header = klamath.library.FileHeader(
        name=library_name.encode('ASCII'),
        user_units_per_db_unit=logical_units_per_unit,
        meters_per_db_unit=meters_per_unit,
        )
    header.write(stream)

    # Now create a structure for each pattern, and add in any Boundary and SREF elements
    for name, pat in library.items():
        elements: List[klamath.elements.Element] = []
        elements += _shapes_to_elements(pat.shapes)
        elements += _labels_to_texts(pat.labels)
        elements += _mrefs_to_grefs(pat.refs)

        klamath.library.write_struct(stream, name=name.encode('ASCII'), elements=elements)
    records.ENDLIB.write(stream, None)


def writefile(
        library: Mapping[str, Pattern],
        filename: Union[str, pathlib.Path],
        *args,
        **kwargs,
        ) -> None:
    """
    Wrapper for `write()` that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.

    Args:
        library: {name: Pattern} pairs to save.
        filename: Filename to save to.
        *args: passed to `write()`
        **kwargs: passed to `write()`
    """
    path = pathlib.Path(filename)

    base_stream = open(path, mode='wb')
    streams: Tuple[Any, ...] = (base_stream,)
    if path.suffix == '.gz':
        stream = cast(BinaryIO, gzip.GzipFile(filename='', mtime=0, fileobj=base_stream))
        streams = (stream,) + streams
    else:
        stream = base_stream

    try:
        write(library, stream, *args, **kwargs)
    finally:
        for ss in streams:
            ss.close()


def readfile(
        filename: Union[str, pathlib.Path],
        *args,
        **kwargs,
        ) -> Tuple[Dict[str, Pattern], Dict[str, Any]]:
    """
    Wrapper for `read()` that takes a filename or path instead of a stream.

    Will automatically decompress gzipped files.

    Args:
        filename: Filename to save to.
        *args: passed to `read()`
        **kwargs: passed to `read()`
    """
    path = pathlib.Path(filename)
    if is_gzipped(path):
        open_func: Callable = gzip.open
    else:
        open_func = open

    with open_func(path, mode='rb') as stream:
        results = read(stream, *args, **kwargs)
    return results


def read(
        stream: BinaryIO,
        raw_mode: bool = True,
        ) -> Tuple[Dict[str, Pattern], Dict[str, Any]]:
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
        - Dict of pattern_name:Patterns generated from GDSII structures
        - Dict of GDSII library info
    """
    library_info = _read_header(stream)

    patterns_dict = {}
    found_struct = records.BGNSTR.skip_past(stream)
    while found_struct:
        name = records.STRNAME.skip_and_read(stream)
        pat = read_elements(stream, raw_mode=raw_mode)
        patterns_dict[name.decode('ASCII')] = pat
        found_struct = records.BGNSTR.skip_past(stream)

    return patterns_dict, library_info


def _read_header(stream: BinaryIO) -> Dict[str, Any]:
    """
    Read the file header and create the library_info dict.
    """
    header = klamath.library.FileHeader.read(stream)

    library_info = {'name': header.name.decode('ASCII'),
                    'meters_per_unit': header.meters_per_db_unit,
                    'logical_units_per_unit': header.user_units_per_db_unit,
                    }
    return library_info


def read_elements(
        stream: BinaryIO,
        raw_mode: bool = True,
        ) -> Pattern:
    """
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

    elements = klamath.library.read_elements(stream)
    for element in elements:
        if isinstance(element, klamath.elements.Boundary):
            poly = _boundary_to_polygon(element, raw_mode)
            pat.shapes.append(poly)
        elif isinstance(element, klamath.elements.Path):
            path = _gpath_to_mpath(element, raw_mode)
            pat.shapes.append(path)
        elif isinstance(element, klamath.elements.Text):
            label = Label(
                offset=element.xy.astype(float),
                layer=element.layer,
                string=element.string.decode('ASCII'),
                annotations=_properties_to_annotations(element.properties),
                )
            pat.labels.append(label)
        elif isinstance(element, klamath.elements.Reference):
            pat.refs.append(_gref_to_mref(element))
    return pat


def _mlayer2gds(mlayer: layer_t) -> Tuple[int, int]:
    """ Helper to turn a layer tuple-or-int into a layer and datatype"""
    if isinstance(mlayer, int):
        layer = mlayer
        data_type = 0
    elif isinstance(mlayer, tuple):
        layer = mlayer[0]
        if len(mlayer) > 1:
            data_type = mlayer[1]
        else:
            data_type = 0
    else:
        raise PatternError(f'Invalid layer for gdsii: {mlayer}. Note that gdsii layers cannot be strings.')
    return layer, data_type


def _gref_to_mref(ref: klamath.library.Reference) -> Ref:
    """
    Helper function to create a Ref from an SREF or AREF. Sets ref.target to struct_name.
    """
    xy = ref.xy.astype(float)
    offset = xy[0]
    repetition = None
    if ref.colrow is not None:
        a_count, b_count = ref.colrow
        a_vector = (xy[1] - offset) / a_count
        b_vector = (xy[2] - offset) / b_count
        repetition = Grid(a_vector=a_vector, b_vector=b_vector,
                          a_count=a_count, b_count=b_count)

    mref = Ref(
        target=ref.struct_name.decode('ASCII'),
        offset=offset,
        rotation=numpy.deg2rad(ref.angle_deg),
        scale=ref.mag,
        mirrored=(ref.invert_y, False),
        annotations=_properties_to_annotations(ref.properties),
        repetition=repetition,
        )
    return mref


def _gpath_to_mpath(gpath: klamath.library.Path, raw_mode: bool) -> Path:
    if gpath.path_type in path_cap_map:
        cap = path_cap_map[gpath.path_type]
    else:
        raise PatternError(f'Unrecognized path type: {gpath.path_type}')

    mpath = Path(
        vertices=gpath.xy.astype(float),
        layer=gpath.layer,
        width=gpath.width,
        cap=cap,
        offset=numpy.zeros(2),
        annotations=_properties_to_annotations(gpath.properties),
        raw=raw_mode,
        )
    if cap == Path.Cap.SquareCustom:
        mpath.cap_extensions = gpath.extension
    return mpath


def _boundary_to_polygon(boundary: klamath.library.Boundary, raw_mode: bool) -> Polygon:
    return Polygon(
        vertices=boundary.xy[:-1].astype(float),
        layer=boundary.layer,
        offset=numpy.zeros(2),
        annotations=_properties_to_annotations(boundary.properties),
        raw=raw_mode,
        )


def _mrefs_to_grefs(refs: List[Ref]) -> List[klamath.library.Reference]:
    grefs = []
    for ref in refs:
        if ref.target is None:
            continue
        encoded_name = ref.target.encode('ASCII')

        # Note: GDS mirrors first and rotates second
        mirror_across_x, extra_angle = normalize_mirror(ref.mirrored)
        rep = ref.repetition
        angle_deg = numpy.rad2deg(ref.rotation + extra_angle) % 360
        properties = _annotations_to_properties(ref.annotations, 512)

        if isinstance(rep, Grid):
            b_vector = rep.b_vector if rep.b_vector is not None else numpy.zeros(2)
            b_count = rep.b_count if rep.b_count is not None else 1
            xy = numpy.array(ref.offset) + numpy.array([
                [0.0, 0.0],
                rep.a_vector * rep.a_count,
                b_vector * b_count,
                ])
            aref = klamath.library.Reference(
                struct_name=encoded_name,
                xy=rint_cast(xy),
                colrow=(numpy.rint(rep.a_count), numpy.rint(rep.b_count)),
                angle_deg=angle_deg,
                invert_y=mirror_across_x,
                mag=ref.scale,
                properties=properties,
                )
            grefs.append(aref)
        elif rep is None:
            sref = klamath.library.Reference(
                struct_name=encoded_name,
                xy=rint_cast([ref.offset]),
                colrow=None,
                angle_deg=angle_deg,
                invert_y=mirror_across_x,
                mag=ref.scale,
                properties=properties,
                )
            grefs.append(sref)
        else:
            new_srefs = [
                klamath.library.Reference(
                    struct_name=encoded_name,
                    xy=rint_cast([ref.offset + dd]),
                    colrow=None,
                    angle_deg=angle_deg,
                    invert_y=mirror_across_x,
                    mag=ref.scale,
                    properties=properties,
                    )
                for dd in rep.displacements]
            grefs += new_srefs
    return grefs


def _properties_to_annotations(properties: Dict[int, bytes]) -> annotations_t:
    return {str(k): [v.decode()] for k, v in properties.items()}


def _annotations_to_properties(annotations: annotations_t, max_len: int = 126) -> Dict[int, bytes]:
    cum_len = 0
    props = {}
    for key, vals in annotations.items():
        try:
            i = int(key)
        except ValueError:
            raise PatternError(f'Annotation key {key} is not convertable to an integer')
        if not (0 < i < 126):
            raise PatternError(f'Annotation key {key} converts to {i} (must be in the range [1,125])')

        val_strings = ' '.join(str(val) for val in vals)
        b = val_strings.encode()
        if len(b) > 126:
            raise PatternError(f'Annotation value {b!r} is longer than 126 characters!')
        cum_len += numpy.ceil(len(b) / 2) * 2 + 2
        if cum_len > max_len:
            raise PatternError(f'Sum of annotation data will be longer than {max_len} bytes! Generated bytes were {b!r}')
        props[i] = b
    return props


def _shapes_to_elements(
        shapes: List[Shape],
        polygonize_paths: bool = False,
        ) -> List[klamath.elements.Element]:
    elements: List[klamath.elements.Element] = []
    # Add a Boundary element for each shape, and Path elements if necessary
    for shape in shapes:
        if shape.repetition is not None:
            raise PatternError('Shape repetitions are not supported by GDS.'
                               ' Please call library.wrap_repeated_shapes() before writing to file.')

        layer, data_type = _mlayer2gds(shape.layer)
        properties = _annotations_to_properties(shape.annotations, 128)
        if isinstance(shape, Path) and not polygonize_paths:
            xy = rint_cast(shape.vertices + shape.offset)
            width = rint_cast(shape.width)
            path_type = next(k for k, v in path_cap_map.items() if v == shape.cap)    # reverse lookup

            extension: Tuple[int, int]
            if shape.cap == Path.Cap.SquareCustom and shape.cap_extensions is not None:
                extension = tuple(shape.cap_extensions)     # type: ignore
            else:
                extension = (0, 0)

            path = klamath.elements.Path(
                layer=(layer, data_type),
                xy=xy,
                path_type=path_type,
                width=int(width),
                extension=extension,
                properties=properties,
                )
            elements.append(path)
        elif isinstance(shape, Polygon):
            polygon = shape
            xy_closed = numpy.empty((polygon.vertices.shape[0] + 1, 2), dtype=numpy.int32)
            numpy.rint(polygon.vertices + polygon.offset, out=xy_closed[:-1], casting='unsafe')
            xy_closed[-1] = xy_closed[0]
            boundary = klamath.elements.Boundary(
                layer=(layer, data_type),
                xy=xy_closed,
                properties=properties,
                )
            elements.append(boundary)
        else:
            for polygon in shape.to_polygons():
                xy_closed = numpy.empty((polygon.vertices.shape[0] + 1, 2), dtype=numpy.int32)
                numpy.rint(polygon.vertices + polygon.offset, out=xy_closed[:-1], casting='unsafe')
                xy_closed[-1] = xy_closed[0]
                boundary = klamath.elements.Boundary(
                    layer=(layer, data_type),
                    xy=xy_closed,
                    properties=properties,
                    )
                elements.append(boundary)
    return elements


def _labels_to_texts(labels: List[Label]) -> List[klamath.elements.Text]:
    texts = []
    for label in labels:
        properties = _annotations_to_properties(label.annotations, 128)
        layer, text_type = _mlayer2gds(label.layer)
        xy = rint_cast([label.offset])
        text = klamath.elements.Text(
            layer=(layer, text_type),
            xy=xy,
            string=label.string.encode('ASCII'),
            properties=properties,
            presentation=0,  # TODO maybe set some of these?
            angle_deg=0,
            invert_y=False,
            width=0,
            path_type=0,
            mag=1,
            )
        texts.append(text)
    return texts


def load_library(
        stream: BinaryIO,
        *,
        full_load: bool = False,
        postprocess: Optional[Callable[[Library, str, Pattern], Pattern]] = None
        ) -> Tuple[LazyLibrary, Dict[str, Any]]:
    """
    Scan a GDSII stream to determine what structures are present, and create
        a library from them. This enables deferred reading of structures
        on an as-needed basis.
    All structures are loaded as secondary

    Args:
        stream: Seekable stream. Position 0 should be the start of the file.
            The caller should leave the stream open while the library
            is still in use, since the library will need to access it
            in order to read the structure contents.
        full_load: If True, force all structures to be read immediately rather
            than as-needed. Since data is read sequentially from the file, this
            will be faster than using the resulting library's `precache` method.
        postprocess: If given, this function is used to post-process each
            pattern *upon first load only*.

    Returns:
        LazyLibrary object, allowing for deferred load of structures.
        Additional library info (dict, same format as from `read`).
    """
    stream.seek(0)
    lib = LazyLibrary()

    if full_load:
        # Full load approach (immediately load everything)
        patterns, library_info = read(stream)
        for name, pattern in patterns.items():
            lib[name] = lambda: pattern
        return lib, library_info

    # Normal approach (scan and defer load)
    library_info = _read_header(stream)
    structs = klamath.library.scan_structs(stream)

    for name_bytes, pos in structs.items():
        name = name_bytes.decode('ASCII')

        def mkstruct(pos: int = pos, name: str = name) -> Pattern:
            logger.error(f'mkstruct {name} @ {pos:x}')
            stream.seek(pos)
            pat = read_elements(stream, raw_mode=True)
            if postprocess is not None:
                pat = postprocess(lib, name, pat)
                logger.error(f'mkstruct post {name} @ {pos:x}')
            return pat

        lib[name] = mkstruct

    return lib, library_info


def load_libraryfile(
        filename: Union[str, pathlib.Path],
        *,
        use_mmap: bool = True,
        full_load: bool = False,
        postprocess: Optional[Callable[[Library, str, Pattern], Pattern]] = None
        ) -> Tuple[LazyLibrary, Dict[str, Any]]:
    """
    Wrapper for `load_library()` that takes a filename or path instead of a stream.

    Will automatically decompress the file if it is gzipped.

    NOTE that any streams/mmaps opened will remain open until ALL of the
     `PatternGenerator` objects in the library are garbage collected.

    Args:
        path: filename or path to read from
        use_mmap: If `True`, will attempt to memory-map the file instead
                  of buffering. In the case of gzipped files, the file
                  is decompressed into a python `bytes` object in memory
                  and reopened as an `io.BytesIO` stream.
        full_load: If `True`, immediately loads all data. See `load_library`.
        postprocess: Passed to `load_library`

    Returns:
        LazyLibrary object, allowing for deferred load of structures.
        Additional library info (dict, same format as from `read`).
    """
    path = pathlib.Path(filename)
    stream: BinaryIO
    if is_gzipped(path):
        if mmap:
            logger.info('Asked to mmap a gzipped file, reading into memory instead...')
            gz_stream = gzip.open(path, mode='rb')
            stream = io.BytesIO(gz_stream.read())       # type: ignore
        else:
            gz_stream = gzip.open(path, mode='rb')
            stream = io.BufferedReader(gz_stream)       # type: ignore
    else:
        if mmap:
            base_stream = open(path, mode='rb', buffering=0)
            stream = mmap.mmap(base_stream.fileno(), 0, access=mmap.ACCESS_READ)    # type: ignore
        else:
            stream = open(path, mode='rb')
    return load_library(stream, full_load=full_load, postprocess=postprocess)


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
