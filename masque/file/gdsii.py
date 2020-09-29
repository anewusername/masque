"""
GDSII file format readers and writers

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
"""
from typing import List, Any, Dict, Tuple, Callable, Union, Sequence, Iterable, Optional
from typing import Sequence, Mapping
import re
import io
import copy
import base64
import struct
import logging
import pathlib
import gzip

import numpy        # type: ignore
# python-gdsii
import gdsii.library
import gdsii.structure
import gdsii.elements

from .utils import mangle_name, make_dose_table, dose2dtype, dtype2dose, clean_pattern_vertices
from .utils import is_gzipped
from .. import Pattern, SubPattern, PatternError, Label, Shape
from ..shapes import Polygon, Path
from ..repetition import Grid
from ..utils import rotation_matrix_2d, get_bit, set_bit, vector2, is_scalar, layer_t
from ..utils import remove_colinear_vertices, normalize_mirror, annotations_t


logger = logging.getLogger(__name__)


path_cap_map = {
                None: Path.Cap.Flush,
                0: Path.Cap.Flush,
                1: Path.Cap.Circle,
                2: Path.Cap.Square,
                4: Path.Cap.SquareCustom,
               }


def build(patterns: Union[Pattern, Sequence[Pattern]],
          meters_per_unit: float,
          logical_units_per_unit: float = 1,
          library_name: str = 'masque-gdsii-write',
          *,
          modify_originals: bool = False,
          disambiguate_func: Callable[[Iterable[Pattern]], None] = None,
          ) -> gdsii.library.Library:
    """
    Convert a `Pattern` or list of patterns to a GDSII stream, by first calling
     `.polygonize()` to change the shapes into polygons, and then writing patterns
     as GDSII structures, polygons as boundary elements, and subpatterns as structure
     references (sref).

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
            Default 'masque-gdsii-write'.
        modify_originals: If `True`, the original pattern is modified as part of the writing
            process. Otherwise, a copy is made and `deepunlock()`-ed.
            Default `False`.
        disambiguate_func: Function which takes a list of patterns and alters them
            to make their names valid and unique. Default is `disambiguate_pattern_names`, which
            attempts to adhere to the GDSII standard as well as possible.
            WARNING: No additional error checking is performed on the results.

    Returns:
        `gdsii.library.Library`
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
    lib = gdsii.library.Library(version=600,
                                name=library_name.encode('ASCII'),
                                logical_unit=logical_units_per_unit,
                                physical_unit=meters_per_unit)

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {id(pattern): pattern for pattern in patterns}
    for pattern in patterns:
        for i, p in pattern.referenced_patterns_by_id().items():
            patterns_by_id[i] = p

    disambiguate_func(patterns_by_id.values())

    # Now create a structure for each pattern, and add in any Boundary and SREF elements
    for pat in patterns_by_id.values():
        structure = gdsii.structure.Structure(name=pat.name.encode('ASCII'))
        lib.append(structure)

        structure += _shapes_to_elements(pat.shapes)
        structure += _labels_to_texts(pat.labels)
        structure += _subpatterns_to_refs(pat.subpatterns)

    return lib


def write(patterns: Union[Pattern, Sequence[Pattern]],
          stream: io.BufferedIOBase,
          *args,
          **kwargs):
    """
    Write a `Pattern` or list of patterns to a GDSII file.
    See `masque.file.gdsii.build()` for details.

    Args:
        patterns: A Pattern or list of patterns to write to file.
        stream: Stream to write to.
        *args: passed to `masque.file.gdsii.build()`
        **kwargs: passed to `masque.file.gdsii.build()`
    """
    lib = build(patterns, *args, **kwargs)
    lib.save(stream)
    return

def writefile(patterns: Union[Sequence[Pattern], Pattern],
              filename: Union[str, pathlib.Path],
              *args,
              **kwargs,
              ):
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
        results = write(patterns, stream, *args, **kwargs)
    return results


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
        results = read(stream, *args, **kwargs)
    return results


def read(stream: io.BufferedIOBase,
         clean_vertices: bool = True,
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
        clean_vertices: If `True`, remove any redundant vertices when loading polygons.
            The cleaning process removes any polygons with zero area or <3 vertices.
            Default `True`.

    Returns:
        - Dict of pattern_name:Patterns generated from GDSII structures
        - Dict of GDSII library info
    """

    lib = gdsii.library.Library.load(stream)

    library_info = {'name': lib.name.decode('ASCII'),
                    'meters_per_unit': lib.physical_unit,
                    'logical_units_per_unit': lib.logical_unit,
                    }

    raw_mode = True     # Whether to construct shapes in raw mode (less error checking)

    patterns = []
    for structure in lib:
        pat = Pattern(name=structure.name.decode('ASCII'))
        for element in structure:
            # Switch based on element type:
            if isinstance(element, gdsii.elements.Boundary):
                poly = _boundary_to_polygon(element, raw_mode)
                pat.shapes.append(poly)

            if isinstance(element, gdsii.elements.Path):
                path = _gpath_to_mpath(element, raw_mode)
                pat.shapes.append(path)

            elif isinstance(element, gdsii.elements.Text):
                label = Label(offset=element.xy.astype(float),
                              layer=(element.layer, element.text_type),
                              string=element.string.decode('ASCII'))
                pat.labels.append(label)

            elif (isinstance(element, gdsii.elements.SRef) or
                  isinstance(element, gdsii.elements.ARef)):
                pat.subpatterns.append(_ref_to_subpat(element))

        if clean_vertices:
            clean_pattern_vertices(pat)
        patterns.append(pat)

    # Create a dict of {pattern.name: pattern, ...}, then fix up all subpattern.pattern entries
    #  according to the subpattern.identifier (which is deleted after use).
    patterns_dict = dict(((p.name, p) for p in patterns))
    for p in patterns_dict.values():
        for sp in p.subpatterns:
            sp.pattern = patterns_dict[sp.identifier[0].decode('ASCII')]
            del sp.identifier

    return patterns_dict, library_info


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


def _ref_to_subpat(element: Union[gdsii.elements.SRef,
                                  gdsii.elements.ARef]
                   ) -> SubPattern:
    """
    Helper function to create a SubPattern from an SREF or AREF. Sets subpat.pattern to None
     and sets the instance .identifier to (struct_name,).

    NOTE: "Absolute" means not affected by parent elements.
           That's not currently supported by masque at all (and not planned).
    """
    rotation = 0.0
    offset = numpy.array(element.xy[0], dtype=float)
    scale = 1.0
    mirror_across_x = False
    repetition = None

    if element.strans is not None:
        if element.mag is not None:
            scale = element.mag
            # Bit 13 means absolute scale
            if get_bit(element.strans, 15 - 13):
                raise PatternError('Absolute scale is not implemented in masque!')
        if element.angle is not None:
            rotation = numpy.deg2rad(element.angle)
            # Bit 14 means absolute rotation
            if get_bit(element.strans, 15 - 14):
                raise PatternError('Absolute rotation is not implemented in masque!')
        # Bit 0 means mirror x-axis
        if get_bit(element.strans, 15 - 0):
            mirror_across_x = True

    if isinstance(element, gdsii.elements.ARef):
        a_count = element.cols
        b_count = element.rows
        a_vector = (element.xy[1] - offset) / a_count
        b_vector = (element.xy[2] - offset) / b_count
        repetition = Grid(a_vector=a_vector, b_vector=b_vector,
                          a_count=a_count, b_count=b_count)

    subpat = SubPattern(pattern=None,
                        offset=offset,
                        rotation=rotation,
                        scale=scale,
                        mirrored=(mirror_across_x, False),
                        annotations=_properties_to_annotations(element.properties),
                        repetition=repetition)
    subpat.identifier = (element.struct_name,)
    return subpat


def _gpath_to_mpath(element: gdsii.elements.Path, raw_mode: bool) -> Path:
    if element.path_type in path_cap_map:
        cap = path_cap_map[element.path_type]
    else:
        raise PatternError(f'Unrecognized path type: {element.path_type}')

    args = {'vertices': element.xy.astype(float),
            'layer': (element.layer, element.data_type),
            'width': element.width if element.width is not None else 0.0,
            'cap': cap,
            'offset': numpy.zeros(2),
            'annotations':_properties_to_annotations(element.properties),
            'raw': raw_mode,
           }

    if cap == Path.Cap.SquareCustom:
        args['cap_extensions'] = numpy.zeros(2)
        if element.bgn_extn is not None:
            args['cap_extensions'][0] = element.bgn_extn
        if element.end_extn is not None:
            args['cap_extensions'][1] = element.end_extn

    return Path(**args)


def _boundary_to_polygon(element: gdsii.elements.Boundary, raw_mode: bool) -> Polygon:
    args = {'vertices': element.xy[:-1].astype(float),
            'layer': (element.layer, element.data_type),
            'offset': numpy.zeros(2),
            'annotations':_properties_to_annotations(element.properties),
            'raw': raw_mode,
           }
    return Polygon(**args)


def _subpatterns_to_refs(subpatterns: List[SubPattern]
                        ) -> List[Union[gdsii.elements.ARef, gdsii.elements.SRef]]:
    refs = []
    for subpat in subpatterns:
        if subpat.pattern is None:
            continue
        encoded_name = subpat.pattern.name.encode('ASCII')

        # Note: GDS mirrors first and rotates second
        mirror_across_x, extra_angle = normalize_mirror(subpat.mirrored)
        rep = subpat.repetition

        new_refs: List[Union[gdsii.elements.SRef, gdsii.elements.ARef]]
        ref: Union[gdsii.elements.SRef, gdsii.elements.ARef]
        if isinstance(rep, Grid):
            xy = numpy.array(subpat.offset) + [
                  [0, 0],
                  rep.a_vector * rep.a_count,
                  rep.b_vector * rep.b_count,
                 ]
            ref = gdsii.elements.ARef(struct_name=encoded_name,
                                       xy=numpy.round(xy).astype(int),
                                       cols=numpy.round(rep.a_count).astype(int),
                                       rows=numpy.round(rep.b_count).astype(int))
            new_refs = [ref]
        elif rep is None:
            ref = gdsii.elements.SRef(struct_name=encoded_name,
                                      xy=numpy.round([subpat.offset]).astype(int))
            new_refs = [ref]
        else:
            new_refs = [gdsii.elements.SRef(struct_name=encoded_name,
                                            xy=numpy.round([subpat.offset + dd]).astype(int))
                        for dd in rep.displacements]

        for ref in new_refs:
            ref.angle = numpy.rad2deg(subpat.rotation + extra_angle) % 360
            #  strans must be non-None for angle and mag to take effect
            ref.strans = set_bit(0, 15 - 0, mirror_across_x)
            ref.mag = subpat.scale
            ref.properties = _annotations_to_properties(subpat.annotations, 512)

        refs += new_refs
    return refs


def _properties_to_annotations(properties: List[Tuple[int, bytes]]) -> annotations_t:
    return {str(k): [v.decode()] for k, v in properties}


def _annotations_to_properties(annotations: annotations_t, max_len: int = 126) -> List[Tuple[int, bytes]]:
    cum_len = 0
    props = []
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
        props.append((i, b))
    return props


def _shapes_to_elements(shapes: List[Shape],
                        polygonize_paths: bool = False
                       ) -> List[Union[gdsii.elements.Boundary, gdsii.elements.Path]]:
    elements: List[Union[gdsii.elements.Boundary, gdsii.elements.Path]] = []
    # Add a Boundary element for each shape, and Path elements if necessary
    for shape in shapes:
        layer, data_type = _mlayer2gds(shape.layer)
        properties = _annotations_to_properties(shape.annotations, 128)
        if isinstance(shape, Path) and not polygonize_paths:
            xy = numpy.round(shape.vertices + shape.offset).astype(int)
            width = numpy.round(shape.width).astype(int)
            path_type = next(k for k, v in path_cap_map.items() if v == shape.cap)    #reverse lookup
            path = gdsii.elements.Path(layer=layer,
                                       data_type=data_type,
                                       xy=xy)
            path.path_type = path_type
            path.width = width
            path.properties = properties
            elements.append(path)
        else:
            for polygon in shape.to_polygons():
                xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                boundary = gdsii.elements.Boundary(layer=layer,
                                                   data_type=data_type,
                                                   xy=xy_closed)
                boundary.properties = properties
                elements.append(boundary)
    return elements


def _labels_to_texts(labels: List[Label]) -> List[gdsii.elements.Text]:
    texts = []
    for label in labels:
        properties = _annotations_to_properties(label.annotations, 128)
        layer, text_type = _mlayer2gds(label.layer)
        xy = numpy.round([label.offset]).astype(int)
        text = gdsii.elements.Text(layer=layer,
                                   text_type=text_type,
                                   xy=xy,
                                   string=label.string.encode('ASCII'))
        text.properties = properties
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
