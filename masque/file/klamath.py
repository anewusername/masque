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
 * Creation/modification/access times are set to 1900-01-01 for reproducibility.
"""
from typing import List, Any, Dict, Tuple, Callable, Union, Sequence, Iterable, Optional
from typing import Sequence, Mapping, BinaryIO
import re
import io
import mmap
import copy
import base64
import struct
import logging
import pathlib
import gzip
from itertools import chain

import numpy        # type: ignore
import klamath
from klamath import records

from .utils import mangle_name, make_dose_table, dose2dtype, dtype2dose, is_gzipped
from .. import Pattern, SubPattern, PatternError, Label, Shape
from ..shapes import Polygon, Path
from ..repetition import Grid
from ..utils import rotation_matrix_2d, get_bit, set_bit, vector2, is_scalar, layer_t
from ..utils import remove_colinear_vertices, normalize_mirror, annotations_t
from ..library import Library

logger = logging.getLogger(__name__)


path_cap_map = {
                0: Path.Cap.Flush,
                1: Path.Cap.Circle,
                2: Path.Cap.Square,
                4: Path.Cap.SquareCustom,
               }


def write(patterns: Union[Pattern, Sequence[Pattern]],
          stream: BinaryIO,
          meters_per_unit: float,
          logical_units_per_unit: float = 1,
          library_name: str = 'masque-klamath',
          *,
          modify_originals: bool = False,
          disambiguate_func: Callable[[Iterable[Pattern]], None] = None,
          ) -> None:
    """
    Convert a `Pattern` or list of patterns to a GDSII stream,  and then mapping data as follows:
         Pattern -> GDSII structure
         SubPattern -> GDSII SREF or AREF
         Path -> GSDII path
         Shape (other than path) -> GDSII boundary/ies
         Label -> GDSII text
         annnotations -> properties, where possible

     For each shape,
        layer is chosen to be equal to `shape.layer` if it is an int,
            or `shape.layer[0]` if it is a tuple
        datatype is chosen to be `shape.layer[1]` if available,
            otherwise `0`

    It is often a good idea to run `pattern.subpatternize()` prior to calling this function,
     especially if calling `.polygonize()` will result in very many vertices.

    If you want pattern polygonized with non-default arguments, just call `pattern.polygonize()`
     prior to calling this function.

    Args:
        patterns: A Pattern or list of patterns to convert.
        meters_per_unit: Written into the GDSII file, meters per (database) length unit.
            All distances are assumed to be an integer multiple of this unit, and are stored as such.
        logical_units_per_unit: Written into the GDSII file. Allows the GDSII to specify a
            "logical" unit which is different from the "database" unit, for display purposes.
            Default `1`.
        library_name: Library name written into the GDSII file.
            Default 'masque-klamath'.
        modify_originals: If `True`, the original pattern is modified as part of the writing
            process. Otherwise, a copy is made and `deepunlock()`-ed.
            Default `False`.
        disambiguate_func: Function which takes a list of patterns and alters them
            to make their names valid and unique. Default is `disambiguate_pattern_names`, which
            attempts to adhere to the GDSII standard as well as possible.
            WARNING: No additional error checking is performed on the results.
    """
    if isinstance(patterns, Pattern):
        patterns = [patterns]

    if disambiguate_func is None:
        disambiguate_func = disambiguate_pattern_names          # type: ignore
    assert(disambiguate_func is not None)       # placate mypy

    if not modify_originals:
        patterns = [p.deepunlock() for p in copy.deepcopy(patterns)]

    patterns = [p.wrap_repeated_shapes() for p in patterns]

    # Create library
    header = klamath.library.FileHeader(name=library_name.encode('ASCII'),
                                        user_units_per_db_unit=logical_units_per_unit,
                                        meters_per_db_unit=meters_per_unit)
    header.write(stream)

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {id(pattern): pattern for pattern in patterns}
    for pattern in patterns:
        for i, p in pattern.referenced_patterns_by_id().items():
            patterns_by_id[i] = p

    disambiguate_func(patterns_by_id.values())

    # Now create a structure for each pattern, and add in any Boundary and SREF elements
    for pat in patterns_by_id.values():
        elements: List[klamath.elements.Element] = []
        elements += _shapes_to_elements(pat.shapes)
        elements += _labels_to_texts(pat.labels)
        elements += _subpatterns_to_refs(pat.subpatterns)

        klamath.library.write_struct(stream, name=pat.name.encode('ASCII'), elements=elements)
    records.ENDLIB.write(stream, None)


def writefile(patterns: Union[Sequence[Pattern], Pattern],
              filename: Union[str, pathlib.Path],
              *args,
              **kwargs,
              ) -> None:
    """
    Wrapper for `masque.file.gdsii.write()` that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.

    Args:
        patterns: `Pattern` or list of patterns to save
        filename: Filename to save to.
        *args: passed to `masque.file.gdsii.write`
        **kwargs: passed to `masque.file.gdsii.write`
    """
    path = pathlib.Path(filename)
    if path.suffix == '.gz':
        open_func: Callable = gzip.open
    else:
        open_func = open

    with io.BufferedWriter(open_func(path, mode='wb')) as stream:
        write(patterns, stream, *args, **kwargs)


def readfile(filename: Union[str, pathlib.Path],
             *args,
             **kwargs,
             ) -> Tuple[Dict[str, Pattern], Dict[str, Any]]:
    """
    Wrapper for `masque.file.gdsii.read()` that takes a filename or path instead of a stream.

    Will automatically decompress gzipped files.

    Args:
        filename: Filename to save to.
        *args: passed to `masque.file.gdsii.read`
        **kwargs: passed to `masque.file.gdsii.read`
    """
    path = pathlib.Path(filename)
    if is_gzipped(path):
        open_func: Callable = gzip.open
    else:
        open_func = open

    with io.BufferedReader(open_func(path, mode='rb')) as stream:
        results = read(stream)#, *args, **kwargs)
    return results


def read(stream: BinaryIO,
         ) -> Tuple[Dict[str, Pattern], Dict[str, Any]]:
    """
    Read a gdsii file and translate it into a dict of Pattern objects. GDSII structures are
     translated into Pattern objects; boundaries are translated into polygons, and srefs and arefs
     are translated into SubPattern objects.

    Additional library info is returned in a dict, containing:
      'name': name of the library
      'meters_per_unit': number of meters per database unit (all values are in database units)
      'logical_units_per_unit': number of "logical" units displayed by layout tools (typically microns)
                                per database unit

    Args:
        stream: Stream to read from.

    Returns:
        - Dict of pattern_name:Patterns generated from GDSII structures
        - Dict of GDSII library info
    """
    raw_mode = True     # Whether to construct shapes in raw mode (less error checking)
    library_info = _read_header(stream)

    patterns = []
    found_struct = records.BGNSTR.skip_past(stream)
    while found_struct:
        name = records.STRNAME.skip_and_read(stream)
        pat = read_elements(stream, name=name.decode('ASCII'))
        patterns.append(pat)
        found_struct = records.BGNSTR.skip_past(stream)

    # Create a dict of {pattern.name: pattern, ...}, then fix up all subpattern.pattern entries
    #  according to the subpattern.identifier (which is deleted after use).
    patterns_dict = dict(((p.name, p) for p in patterns))
    for p in patterns_dict.values():
        for sp in p.subpatterns:
            sp.pattern = patterns_dict[sp.identifier[0]]
            del sp.identifier

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


def read_elements(stream: BinaryIO,
                  name: str,
                  raw_mode: bool = True,
                  ) -> Pattern:
    """
    Read elements from a GDS structure and build a Pattern from them.

    Args:
        stream: Seekable stream, positioned at a record boundary.
                Will be read until an ENDSTR record is consumed.
        name: Name of the resulting Pattern
        raw_mode: If True, bypass per-shape consistency checking

    Returns:
        A pattern containing the elements that were read.
    """
    pat = Pattern(name)

    elements = klamath.library.read_elements(stream)
    for element in elements:
        if isinstance(element, klamath.elements.Boundary):
            poly = _boundary_to_polygon(element, raw_mode)
            pat.shapes.append(poly)
        elif isinstance(element, klamath.elements.Path):
            path = _gpath_to_mpath(element, raw_mode)
            pat.shapes.append(path)
        elif isinstance(element, klamath.elements.Text):
            label = Label(offset=element.xy.astype(float),
                          layer=element.layer,
                          string=element.string.decode('ASCII'),
                          annotations=_properties_to_annotations(element.properties))
            pat.labels.append(label)
        elif isinstance(element, klamath.elements.Reference):
            pat.subpatterns.append(_ref_to_subpat(element))
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


def _ref_to_subpat(ref: klamath.library.Reference,
                   ) -> SubPattern:
    """
    Helper function to create a SubPattern from an SREF or AREF. Sets subpat.pattern to None
     and sets the instance .identifier to (struct_name,).
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

    subpat = SubPattern(pattern=None,
                        offset=offset,
                        rotation=numpy.deg2rad(ref.angle_deg),
                        scale=ref.mag,
                        mirrored=(ref.invert_y, False),
                        annotations=_properties_to_annotations(ref.properties),
                        repetition=repetition)
    subpat.identifier = (ref.struct_name.decode('ASCII'),)
    return subpat


def _gpath_to_mpath(gpath: klamath.library.Path, raw_mode: bool) -> Path:
    if gpath.path_type in path_cap_map:
        cap = path_cap_map[gpath.path_type]
    else:
        raise PatternError(f'Unrecognized path type: {gpath.path_type}')

    mpath = Path(vertices=gpath.xy.astype(float),
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
    return Polygon(vertices=boundary.xy[:-1].astype(float),
                   layer=boundary.layer,
                   offset=numpy.zeros(2),
                   annotations=_properties_to_annotations(boundary.properties),
                   raw=raw_mode,
                   )


def _subpatterns_to_refs(subpatterns: List[SubPattern]
                        ) -> List[klamath.library.Reference]:
    refs = []
    for subpat in subpatterns:
        if subpat.pattern is None:
            continue
        encoded_name = subpat.pattern.name.encode('ASCII')

        # Note: GDS mirrors first and rotates second
        mirror_across_x, extra_angle = normalize_mirror(subpat.mirrored)
        rep = subpat.repetition
        angle_deg = numpy.rad2deg(subpat.rotation + extra_angle) % 360
        properties = _annotations_to_properties(subpat.annotations, 512)

        if isinstance(rep, Grid):
            xy = numpy.array(subpat.offset) + [
                  [0, 0],
                  rep.a_vector * rep.a_count,
                  rep.b_vector * rep.b_count,
                 ]
            aref = klamath.library.Reference(struct_name=encoded_name,
                                             xy=numpy.round(xy).astype(int),
                                             colrow=(numpy.round(rep.a_count), numpy.round(rep.b_count)),
                                             angle_deg=angle_deg,
                                             invert_y=mirror_across_x,
                                             mag=subpat.scale,
                                             properties=properties)
            refs.append(aref)
        elif rep is None:
            ref = klamath.library.Reference(struct_name=encoded_name,
                                            xy=numpy.round([subpat.offset]).astype(int),
                                            colrow=None,
                                            angle_deg=angle_deg,
                                            invert_y=mirror_across_x,
                                            mag=subpat.scale,
                                            properties=properties)
            refs.append(ref)
        else:
            new_srefs = [klamath.library.Reference(struct_name=encoded_name,
                                                   xy=numpy.round([subpat.offset + dd]).astype(int),
                                                   colrow=None,
                                                   angle_deg=angle_deg,
                                                   invert_y=mirror_across_x,
                                                   mag=subpat.scale,
                                                   properties=properties)
                         for dd in rep.displacements]
            refs += new_srefs
    return refs


def _properties_to_annotations(properties: Dict[int, bytes]) -> annotations_t:
    return {str(k): [v.decode()] for k, v in properties.items()}


def _annotations_to_properties(annotations: annotations_t, max_len: int = 126) -> Dict[int, bytes]:
    cum_len = 0
    props = {}
    for key, vals in annotations.items():
        try:
            i = int(key)
        except:
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


def _shapes_to_elements(shapes: List[Shape],
                        polygonize_paths: bool = False
                       ) -> List[klamath.elements.Element]:
    elements: List[klamath.elements.Element] = []
    # Add a Boundary element for each shape, and Path elements if necessary
    for shape in shapes:
        layer, data_type = _mlayer2gds(shape.layer)
        properties = _annotations_to_properties(shape.annotations, 128)
        if isinstance(shape, Path) and not polygonize_paths:
            xy = numpy.round(shape.vertices + shape.offset).astype(int)
            width = numpy.round(shape.width).astype(int)
            path_type = next(k for k, v in path_cap_map.items() if v == shape.cap)    #reverse lookup

            extension: Tuple[int, int]
            if shape.cap == Path.Cap.SquareCustom and shape.cap_extensions is not None:
                extension = tuple(shape.cap_extensions)     # type: ignore
            else:
                extension = (0, 0)

            path = klamath.elements.Path(layer=(layer, data_type),
                                         xy=xy,
                                         path_type=path_type,
                                         width=width,
                                         extension=extension,
                                         properties=properties)
            elements.append(path)
        elif isinstance(shape, Polygon):
           polygon = shape
           xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
           xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
           boundary = klamath.elements.Boundary(layer=(layer, data_type),
                                                xy=xy_closed,
                                                properties=properties)
           elements.append(boundary)
        else:
            for polygon in shape.to_polygons():
                xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                boundary = klamath.elements.Boundary(layer=(layer, data_type),
                                                     xy=xy_closed,
                                                     properties=properties)
                elements.append(boundary)
    return elements


def _labels_to_texts(labels: List[Label]) -> List[klamath.elements.Text]:
    texts = []
    for label in labels:
        properties = _annotations_to_properties(label.annotations, 128)
        layer, text_type = _mlayer2gds(label.layer)
        xy = numpy.round([label.offset]).astype(int)
        text = klamath.elements.Text(layer=(layer, text_type),
                                     xy=xy,
                                     string=label.string.encode('ASCII'),
                                     properties=properties,
                                     presentation=0, #TODO maybe set some of these?
                                     angle_deg=0,
                                     invert_y=False,
                                     width=0,
                                     path_type=0,
                                     mag=1)
        texts.append(text)
    return texts


def disambiguate_pattern_names(patterns: Sequence[Pattern],
                               max_name_length: int = 32,
                               suffix_length: int = 6,
                               dup_warn_filter: Optional[Callable[[str,], bool]] = None,
                               ):
    """
    Args:
        patterns: List of patterns to disambiguate
        max_name_length: Names longer than this will be truncated
        suffix_length: Names which get truncated are truncated by this many extra characters. This is to
            leave room for a suffix if one is necessary.
        dup_warn_filter: (optional) Function for suppressing warnings about cell names changing. Receives
            the cell name and returns `False` if the warning should be suppressed and `True` if it should
            be displayed. Default displays all warnings.
    """
    used_names = []
    for pat in set(patterns):
        # Shorten names which already exceed max-length
        if len(pat.name) > max_name_length:
            shortened_name = pat.name[:max_name_length - suffix_length]
            logger.warning(f'Pattern name "{pat.name}" is too long ({len(pat.name)}/{max_name_length} chars),\n' +
                           f' shortening to "{shortened_name}" before generating suffix')
        else:
            shortened_name = pat.name

        # Remove invalid characters
        sanitized_name = re.compile('[^A-Za-z0-9_\?\$]').sub('_', shortened_name)

        # Add a suffix that makes the name unique
        i = 0
        suffixed_name = sanitized_name
        while suffixed_name in used_names or suffixed_name == '':
            suffix = base64.b64encode(struct.pack('>Q', i), b'$?').decode('ASCII')

            suffixed_name = sanitized_name + '$' + suffix[:-1].lstrip('A')
            i += 1

        if sanitized_name == '':
            logger.warning(f'Empty pattern name saved as "{suffixed_name}"')
        elif suffixed_name != sanitized_name:
            if dup_warn_filter is None or dup_warn_filter(pat.name):
                logger.warning(f'Pattern name "{pat.name}" ({sanitized_name}) appears multiple times;\n' +
                               f' renaming to "{suffixed_name}"')

        # Encode into a byte-string and perform some final checks
        encoded_name = suffixed_name.encode('ASCII')
        if len(encoded_name) == 0:
            # Should never happen since zero-length names are replaced
            raise PatternError(f'Zero-length name after sanitize+encode,\n originally "{pat.name}"')
        if len(encoded_name) > max_name_length:
            raise PatternError(f'Pattern name "{encoded_name!r}" length > {max_name_length} after encode,\n' +
                               f' originally "{pat.name}"')

        pat.name = suffixed_name
        used_names.append(suffixed_name)


def load_library(stream: BinaryIO,
                 tag: str,
                 is_secondary: Optional[Callable[[str], bool]] = None,
                 ) -> Tuple[Library, Dict[str, Any]]:
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
        tag: Unique identifier that will be used to identify this data source
        is_secondary: Function which takes a structure name and returns
                      True if the structure should only be used as a subcell
                      and not appear in the main Library interface.
                      Default always returns False.

    Returns:
        Library object, allowing for deferred load of structures.
        Additional library info (dict, same format as from `read`).
    """
    if is_secondary is None:
        is_secondary = lambda k: False

    stream.seek(0)
    library_info = _read_header(stream)
    structs = klamath.library.scan_structs(stream)

    lib = Library()
    for name_bytes, pos in structs.items():
        name = name_bytes.decode('ASCII')

        def mkstruct(pos: int = pos, name: str = name) -> Pattern:
            stream.seek(pos)
            return read_elements(stream, name, raw_mode=True)

        lib.set_value(name, tag, mkstruct, secondary=is_secondary(name))

    return lib


def load_libraryfile(filename: Union[str, pathlib.Path],
                     tag: str,
                     is_secondary: Optional[Callable[[str], bool]] = None,
                     use_mmap: bool = True,
                     ) -> Tuple[Library, Dict[str, Any]]:
    """
    Wrapper for `load_library()` that takes a filename or path instead of a stream.

    Will automatically decompress the file if it is gzipped.

    NOTE that any streams/mmaps opened will remain open until ALL of the
     `PatternGenerator` objects in the library are garbage collected.

    Args:
        path: filename or path to read from
        tag: Unique identifier for library, see `load_library`
        is_secondary: Function specifying subcess, see `load_library`
        use_mmap: If `True`, will attempt to memory-map the file instead
                  of buffering. In the case of gzipped files, the file
                  is decompressed into a python `bytes` object in memory
                  and reopened as an `io.BytesIO` stream.

    Returns:
        Library object, allowing for deferred load of structures.
        Additional library info (dict, same format as from `read`).
    """
    path = pathlib.Path(filename)
    if is_gzipped(path):
        if mmap:
            logger.info('Asked to mmap a gzipped file, reading into memory instead...')
            base_stream = gzip.open(path, mode='rb')
            stream = io.BytesIO(base_stream.read())
        else:
            base_stream = gzip.open(path, mode='rb')
            stream = io.BufferedReader(base_stream)
    else:
        base_stream = open(path, mode='rb')
        if mmap:
            stream = mmap.mmap(base_stream.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            stream = io.BufferedReader(base_stream)
    return load_library(stream, tag, is_secondary)
