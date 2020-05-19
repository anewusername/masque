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
"""
from typing import List, Any, Dict, Tuple, Callable, Union, Sequence, Iterable, Optional
import re
import io
import copy
import numpy
import base64
import struct
import logging
import pathlib
import gzip

# python-gdsii
import gdsii.library
import gdsii.structure
import gdsii.elements

from .utils import mangle_name, make_dose_table, dose2dtype, dtype2dose
from .. import Pattern, SubPattern, GridRepetition, PatternError, Label, Shape, subpattern_t
from ..shapes import Polygon, Path
from ..utils import rotation_matrix_2d, get_bit, set_bit, vector2, is_scalar, layer_t
from ..utils import remove_colinear_vertices, normalize_mirror

#TODO absolute positioning


logger = logging.getLogger(__name__)


path_cap_map = {
                None: Path.Cap.Flush,
                0: Path.Cap.Flush,
                1: Path.Cap.Circle,
                2: Path.Cap.Square,
                4: Path.Cap.SquareCustom,
               }


def build(patterns: Union[Pattern, List[Pattern]],
          meters_per_unit: float,
          logical_units_per_unit: float = 1,
          library_name: str = 'masque-gdsii-write',
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
        disambiguate_func = disambiguate_pattern_names

    if not modify_originals:
        patterns = [p.deepunlock() for p in copy.deepcopy(patterns)]

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
        structure = gdsii.structure.Structure(name=pat.name)
        lib.append(structure)

        structure += _shapes_to_elements(pat.shapes)
        structure += _labels_to_texts(pat.labels)
        structure += _subpatterns_to_refs(pat.subpatterns)

    return lib


def write(patterns: Union[Pattern, List[Pattern]],
          stream: io.BufferedIOBase,
          *args,
          **kwargs):
    """
    Write a `Pattern` or list of patterns to a GDSII file.
    See `masque.file.gdsii.build()` for details.

    Args:
        patterns: A Pattern or list of patterns to write to file.
        stream: Stream to write to.
        *args: passed to `oasis.build()`
        **kwargs: passed to `oasis.build()`
    """
    lib = build(patterns, *args, **kwargs)
    lib.save(stream)
    return

def writefile(patterns: Union[List[Pattern], Pattern],
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

    Will automatically decompress files with a .gz suffix.

    Args:
        filename: Filename to save to.
        *args: passed to `masque.file.gdsii.read`
        **kwargs: passed to `masque.file.gdsii.read`
    """
    path = pathlib.Path(filename)
    if path.suffix == '.gz':
        open_func: Callable = gzip.open
    else:
        open_func = open

    with io.BufferedReader(open_func(path, mode='rb')) as stream:
        results = read(stream, *args, **kwargs)
    return results


def read(stream: io.BufferedIOBase,
         use_dtype_as_dose: bool = False,
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
        use_dtype_as_dose: If `False`, set each polygon's layer to `(gds_layer, gds_datatype)`.
            If `True`, set the layer to `gds_layer` and the dose to `gds_datatype`.
            Default `False`.
            NOTE: This will be deprecated in the future in favor of
                    `pattern.apply(masque.file.utils.dtype2dose)`.
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

    patterns = []
    for structure in lib:
        pat = Pattern(name=structure.name.decode('ASCII'))
        for element in structure:
            # Switch based on element type:
            if isinstance(element, gdsii.elements.Boundary):
                args = {'vertices': element.xy[:-1],
                        'layer': (element.layer, element.data_type),
                       }

                poly = Polygon(**args)

                if clean_vertices:
                    try:
                        poly.clean_vertices()
                    except PatternError:
                        continue

                pat.shapes.append(poly)

            if isinstance(element, gdsii.elements.Path):
                if element.path_type in path_cap_map:
                    cap = path_cap_map[element.path_type]
                else:
                    raise PatternError('Unrecognized path type: {}'.format(element.path_type))

                args = {'vertices': element.xy,
                        'layer': (element.layer, element.data_type),
                        'width': element.width if element.width is not None else 0.0,
                        'cap': cap,
                       }

                if cap == Path.Cap.SquareCustom:
                    args['cap_extensions'] = numpy.zeros(2)
                    if element.bgn_extn is not None:
                        args['cap_extensions'][0] = element.bgn_extn
                    if element.end_extn is not None:
                        args['cap_extensions'][1] = element.end_extn

                path = Path(**args)

                if clean_vertices:
                    try:
                        path.clean_vertices()
                    except PatternError as err:
                        continue

                pat.shapes.append(path)

            elif isinstance(element, gdsii.elements.Text):
                label = Label(offset=element.xy,
                              layer=(element.layer, element.text_type),
                              string=element.string.decode('ASCII'))
                pat.labels.append(label)

            elif isinstance(element, gdsii.elements.SRef):
                pat.subpatterns.append(_sref_to_subpat(element))

            elif isinstance(element, gdsii.elements.ARef):
                pat.subpatterns.append(_aref_to_gridrep(element))

        if use_dtype_as_dose:
            logger.warning('use_dtype_as_dose will be removed in the future!')
            pat = dtype2dose(pat)

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
        raise PatternError(f'Invalid layer for gdsii: {layer}. Note that gdsii layers cannot be strings.')
    return layer, data_type


def _sref_to_subpat(element: gdsii.elements.SRef) -> SubPattern:
    """
    Helper function to create a SubPattern from an SREF. Sets subpat.pattern to None
     and sets the instance .identifier to (struct_name,).

    BUG:
        "Absolute" means not affected by parent elements.
          That's not currently supported by masque at all, so need to either tag it and
          undo the parent transformations, or implement it in masque.
    """
    subpat = SubPattern(pattern=None, offset=element.xy)
    subpat.identifier = (element.struct_name,)
    if element.strans is not None:
        if element.mag is not None:
            subpat.scale = element.mag
            # Bit 13 means absolute scale
            if get_bit(element.strans, 15 - 13):
                #subpat.offset *= subpat.scale
                raise PatternError('Absolute scale is not implemented yet!')
        if element.angle is not None:
            subpat.rotation = element.angle * numpy.pi / 180
            # Bit 14 means absolute rotation
            if get_bit(element.strans, 15 - 14):
                #subpat.offset = numpy.dot(rotation_matrix_2d(subpat.rotation), subpat.offset)
                raise PatternError('Absolute rotation is not implemented yet!')
        # Bit 0 means mirror x-axis
        if get_bit(element.strans, 15 - 0):
            subpat.mirrored[0] = 1
    return subpat


def _aref_to_gridrep(element: gdsii.elements.ARef) -> GridRepetition:
    """
    Helper function to create a GridRepetition from an AREF. Sets gridrep.pattern to None
     and sets the instance .identifier to (struct_name,).

    BUG:
        "Absolute" means not affected by parent elements.
          That's not currently supported by masque at all, so need to either tag it and
          undo the parent transformations, or implement it in masque.
    """
    rotation = 0
    offset = numpy.array(element.xy[0])
    scale = 1
    mirror_across_x = False

    if element.strans is not None:
        if element.mag is not None:
            scale = element.mag
            # Bit 13 means absolute scale
            if get_bit(element.strans, 15 - 13):
                raise PatternError('Absolute scale is not implemented yet!')
        if element.angle is not None:
            rotation = element.angle * numpy.pi / 180
            # Bit 14 means absolute rotation
            if get_bit(element.strans, 15 - 14):
                raise PatternError('Absolute rotation is not implemented yet!')
        # Bit 0 means mirror x-axis
        if get_bit(element.strans, 15 - 0):
            mirror_across_x = True

    counts = [element.cols, element.rows]
    a_vector = (element.xy[1] - offset) / counts[0]
    b_vector = (element.xy[2] - offset) / counts[1]

    gridrep = GridRepetition(pattern=None,
                            a_vector=a_vector,
                            b_vector=b_vector,
                            a_count=counts[0],
                            b_count=counts[1],
                            offset=offset,
                            rotation=rotation,
                            scale=scale,
                            mirrored=(mirror_across_x, False))
    gridrep.identifier = (element.struct_name,)

    return gridrep


def _subpatterns_to_refs(subpatterns: List[subpattern_t]
                        ) -> List[Union[gdsii.elements.ARef, gdsii.elements.SRef]]:
    refs = []
    for subpat in subpatterns:
        if subpat.pattern is None:
            continue
        encoded_name = subpat.pattern.name

        # Note: GDS mirrors first and rotates second
        mirror_across_x, extra_angle = normalize_mirror(subpat.mirrored)
        ref: Union[gdsii.elements.SRef, gdsii.elements.ARef]
        if isinstance(subpat, GridRepetition):
            xy = numpy.array(subpat.offset) + [
                  [0, 0],
                  subpat.a_vector * subpat.a_count,
                  subpat.b_vector * subpat.b_count,
                 ]
            ref = gdsii.elements.ARef(struct_name=encoded_name,
                                       xy=numpy.round(xy).astype(int),
                                       cols=numpy.round(subpat.a_count).astype(int),
                                       rows=numpy.round(subpat.b_count).astype(int))
        else:
            ref = gdsii.elements.SRef(struct_name=encoded_name,
                                      xy=numpy.round([subpat.offset]).astype(int))

        ref.angle = ((subpat.rotation + extra_angle) * 180 / numpy.pi) % 360
        #  strans must be non-None for angle and mag to take effect
        ref.strans = set_bit(0, 15 - 0, mirror_across_x)
        ref.mag = subpat.scale

        refs.append(ref)
    return refs


def _shapes_to_elements(shapes: List[Shape],
                        polygonize_paths: bool = False
                       ) -> List[Union[gdsii.elements.Boundary, gdsii.elements.Path]]:
    elements: List[Union[gdsii.elements.Boundary, gdsii.elements.Path]] = []
    # Add a Boundary element for each shape, and Path elements if necessary
    for shape in shapes:
        layer, data_type = _mlayer2gds(shape.layer)
        if isinstance(shape, Path) and not polygonize_paths:
            xy = numpy.round(shape.vertices + shape.offset).astype(int)
            width = numpy.round(shape.width).astype(int)
            path_type = next(k for k, v in path_cap_map.items() if v == shape.cap)    #reverse lookup
            path = gdsii.elements.Path(layer=layer,
                                       data_type=data_type,
                                       xy=xy)
            path.path_type = path_type
            path.width = width
            elements.append(path)
        else:
            for polygon in shape.to_polygons():
                xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                elements.append(gdsii.elements.Boundary(layer=layer,
                                                        data_type=data_type,
                                                        xy=xy_closed))
    return elements


def _labels_to_texts(labels: List[Label]) -> List[gdsii.elements.Text]:
    texts = []
    for label in labels:
        layer, text_type = _mlayer2gds(label.layer)
        xy = numpy.round([label.offset]).astype(int)
        texts.append(gdsii.elements.Text(layer=layer,
                                         text_type=text_type,
                                         xy=xy,
                                         string=label.string.encode('ASCII')))
    return texts


def disambiguate_pattern_names(patterns,
                               max_name_length: int = 32,
                               suffix_length: int = 6,
                               dup_warn_filter: Callable[[str,], bool] = None,      # If returns False, don't warn about this name
                               ):
    used_names = []
    for pat in patterns:
        if len(pat.name) > max_name_length:
            shortened_name = pat.name[:max_name_length - suffix_length]
            logger.warning('Pattern name "{}" is too long ({}/{} chars),\n'.format(pat.name, len(pat.name), max_name_length) +
                           ' shortening to "{}" before generating suffix'.format(shortened_name))
        else:
            shortened_name = pat.name

        sanitized_name = re.compile('[^A-Za-z0-9_\?\$]').sub('_', shortened_name)

        i = 0
        suffixed_name = sanitized_name
        while suffixed_name in used_names or suffixed_name == '':
            suffix = base64.b64encode(struct.pack('>Q', i), b'$?').decode('ASCII')

            suffixed_name = sanitized_name + '$' + suffix[:-1].lstrip('A')
            i += 1

        if sanitized_name == '':
            logger.warning('Empty pattern name saved as "{}"'.format(suffixed_name))
        elif suffixed_name != sanitized_name:
            if dup_warn_filter is None or dup_warn_filter(pat.name):
                logger.warning('Pattern name "{}" ({}) appears multiple times;\n renaming to "{}"'.format(
                                pat.name, sanitized_name, suffixed_name))

        encoded_name = suffixed_name.encode('ASCII')
        if len(encoded_name) == 0:
            # Should never happen since zero-length names are replaced
            raise PatternError('Zero-length name after sanitize+encode,\n originally "{}"'.format(pat.name))
        if len(encoded_name) > max_name_length:
            raise PatternError('Pattern name "{!r}" length > {} after encode,\n originally "{}"'.format(encoded_name, max_name_length, pat.name))

        pat.name = encoded_name
        used_names.append(suffixed_name)
