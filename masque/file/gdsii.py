"""
GDSII file format readers and writers
"""
# python-gdsii
import gdsii.library
import gdsii.structure
import gdsii.elements

from typing import List, Any, Dict, Tuple
import re
import io
import copy
import numpy
import base64
import struct
import logging
import pathlib
import gzip

from .utils import mangle_name, make_dose_table
from .. import Pattern, SubPattern, GridRepetition, PatternError, Label, Shape
from ..shapes import Polygon, Path
from ..utils import rotation_matrix_2d, get_bit, set_bit, vector2, is_scalar
from ..utils import remove_colinear_vertices


__author__ = 'Jan Petykiewicz'


logger = logging.getLogger(__name__)


path_cap_map = {
                None: Path.Cap.Flush,
                0: Path.Cap.Flush,
                1: Path.Cap.Circle,
                2: Path.Cap.Square,
               #3: custom?
               }


def write(patterns: Pattern or List[Pattern],
          stream: io.BufferedIOBase,
          meters_per_unit: float,
          logical_units_per_unit: float = 1,
          library_name: str = 'masque-gdsii-write',
          modify_originals: bool = False):
    """
    Write a Pattern or list of patterns to a GDSII file, by first calling
     .polygonize() to change the shapes into polygons, and then writing patterns
     as GDSII structures, polygons as boundary elements, and subpatterns as structure
     references (sref).

     For each shape,
        layer is chosen to be equal to shape.layer if it is an int,
            or shape.layer[0] if it is a tuple
        datatype is chosen to be shape.layer[1] if available,
            otherwise 0

    It is often a good idea to run pattern.subpatternize() prior to calling this function,
     especially if calling .polygonize() will result in very many vertices.

    If you want pattern polygonized with non-default arguments, just call pattern.polygonize()
     prior to calling this function.

    :param patterns: A Pattern or list of patterns to write to file.
    :param file: Filename or stream object to write to.
    :param meters_per_unit: Written into the GDSII file, meters per (database) length unit.
        All distances are assumed to be an integer multiple of this unit, and are stored as such.
    :param logical_units_per_unit: Written into the GDSII file. Allows the GDSII to specify a
        "logical" unit which is different from the "database" unit, for display purposes.
        Default 1.
    :param library_name: Library name written into the GDSII file.
        Default 'masque-gdsii-write'.
    :param modify_originals: If True, the original pattern is modified as part of the writing
        process. Otherwise, a copy is made.
        Default False.
    """
    if not modify_originals:
        patterns = copy.deepcopy(patterns)

    if isinstance(patterns, Pattern):
        patterns = [patterns]

    # Create library
    lib = gdsii.library.Library(version=600,
                                name=library_name.encode('ASCII'),
                                logical_unit=logical_units_per_unit,
                                physical_unit=meters_per_unit)

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {id(pattern): pattern for pattern in patterns}
    for pattern in patterns:
        patterns_by_id.update(pattern.referenced_patterns_by_id())

    _disambiguate_pattern_names(patterns_by_id.values())

    # Now create a structure for each pattern, and add in any Boundary and SREF elements
    for pat in patterns_by_id.values():
        structure = gdsii.structure.Structure(name=pat.name)
        lib.append(structure)

        structure += _shapes_to_elements(pat.shapes)
        structure += _labels_to_texts(pat.labels)
        structure += _subpatterns_to_refs(pat.subpatterns)

    lib.save(stream)
    return


def writefile(patterns: List[Pattern] or Pattern,
              filename: str or pathlib.Path,
              *args,
              **kwargs,
              ):
    """
    Wrapper for gdsii.write() that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.
    """
    path = pathlib.Path(filename)
    if path.suffix == '.gz':
        open_func = gzip.open
    else:
        open_func = open

    with open_func(path, mode='wb') as stream:
        results = write(patterns, stream, *args, **kwargs)
    return results


def dose2dtype(patterns: Pattern or List[Pattern],
              ) -> Tuple[List[Pattern], List[float]]:
    """
     For each shape in each pattern, set shape.layer to the tuple
     (base_layer, datatype), where:
        layer is chosen to be equal to the original shape.layer if it is an int,
            or shape.layer[0] if it is a tuple
        datatype is chosen arbitrarily, based on calcualted dose for each shape.
            Shapes with equal calcualted dose will have the same datatype.
            A list of doses is retured, providing a mapping between datatype
            (list index) and dose (list entry).

    Note that this function modifies the input Pattern(s).

    :param patterns: A Pattern or list of patterns to write to file. Modified by this function.
    :returns: (patterns, dose_list)
            patterns: modified input patterns
            dose_list: A list of doses, providing a mapping between datatype (int, list index)
                       and dose (float, list entry).
    """
    if isinstance(patterns, Pattern):
        patterns = [patterns]

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {id(pattern): pattern for pattern in patterns}
    for pattern in patterns:
        patterns_by_id.update(pattern.referenced_patterns_by_id())

    # Get a table of (id(pat), written_dose) for each pattern and subpattern
    sd_table = make_dose_table(patterns)

    # Figure out all the unique doses necessary to write this pattern
    #  This means going through each row in sd_table and adding the dose values needed to write
    #  that subpattern at that dose level
    dose_vals = set()
    for pat_id, pat_dose in sd_table:
        pat = patterns_by_id[pat_id]
        [dose_vals.add(shape.dose * pat_dose) for shape in pat.shapes]

    if len(dose_vals) > 256:
        raise PatternError('Too many dose values: {}, maximum 256 when using dtypes.'.format(len(dose_vals)))

    # Create a new pattern for each non-1-dose entry in the dose table
    #   and update the shapes to reflect their new dose
    new_pats = {} # (id, dose) -> new_pattern mapping
    for pat_id, pat_dose in sd_table:
        if pat_dose == 1:
            new_pats[(pat_id, pat_dose)] = patterns_by_id[pat_id]
            continue

        pat = patterns_by_id[pat_id].deepcopy()

        encoded_name = mangle_name(pat, pat_dose).encode('ASCII')
        if len(encoded_name) == 0:
            raise PatternError('Zero-length name after mangle+encode, originally "{}"'.format(pat.name))

        for shape in pat.shapes:
            data_type = dose_vals_list.index(shape.dose * pat_dose)
            if is_scalar(shape.layer):
                layer = (shape.layer, data_type)
            else:
                layer = (shape.layer[0], data_type)

        new_pats[(pat_id, pat_dose)] = pat

    # Go back through all the dose-specific patterns and fix up their subpattern entries
    for (pat_id, pat_dose), pat in new_pats.items():
        for subpat in pat.subpatterns:
            dose_mult = subpat.dose * pat_dose
            subpat.pattern = new_pats[(id(subpat.pattern), dose_mult)]

    return patterns, list(dose_vals)


def readfile(filename: str or pathlib.Path,
             *args,
             **kwargs,
             ) -> (Dict[str, Pattern], Dict[str, Any]):
    """
    Wrapper for gdsii.read() that takes a filename or path instead of a stream.

    Tries to autodetermine file type based on suffixes
    """
    path = pathlib.Path(filename)
    if path.suffix == '.gz':
        open_func = gzip.open
    else:
        open_func = open

    with open_func(path, mode='rb') as stream:
        results = read(stream, *args, **kwargs)
    return results


def read(stream: io.BufferedIOBase,
         use_dtype_as_dose: bool = False,
         clean_vertices: bool = True,
         ) -> (Dict[str, Pattern], Dict[str, Any]):
    """
    Read a gdsii file and translate it into a dict of Pattern objects. GDSII structures are
     translated into Pattern objects; boundaries are translated into polygons, and srefs and arefs
     are translated into SubPattern objects.

    Additional library info is returned in a dict, containing:
      'name': name of the library
      'meters_per_unit': number of meters per database unit (all values are in database units)
      'logical_units_per_unit': number of "logical" units displayed by layout tools (typically microns)
                                per database unit

    :param filename: Filename specifying a GDSII file to read from.
    :param use_dtype_as_dose: If false, set each polygon's layer to (gds_layer, gds_datatype).
            If true, set the layer to gds_layer and the dose to gds_datatype.
            Default False.
    :param clean_vertices: If true, remove any redundant vertices when loading polygons.
            The cleaning process removes any polygons with zero area or <3 vertices.
            Default True.
    :return: Tuple: (Dict of pattern_name:Patterns generated from GDSII structures, Dict of GDSII library info)
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
                       }

                if use_dtype_as_dose:
                    args['dose'] = element.data_type
                    args['layer'] = element.layer
                else:
                    args['layer'] = (element.layer, element.data_type)

                shape = Polygon(**args)

                if clean_vertices:
                    try:
                        shape.clean_vertices()
                    except PatternError:
                        continue

                pat.shapes.append(shape)

            if isinstance(element, gdsii.elements.Path):
                if element.path_type in path_cap_map:
                    cap = path_cap_map[element.path_type]
                else:
                    raise PatternError('Unrecognized path type: {}'.format(element.path_type))

                args = {'vertices': element.xy,
                        'width': element.width if element.width is not None else 0.0,
                        'cap': cap,
                       }

                if use_dtype_as_dose:
                    args['dose'] = element.data_type
                    args['layer'] = element.layer
                else:
                    args['layer'] = (element.layer, element.data_type)

                shape = Path(**args)

                if clean_vertices:
                    try:
                        shape.clean_vertices()
                    except PatternError as err:
                        continue

                pat.shapes.append(shape)

            elif isinstance(element, gdsii.elements.Text):
                label = Label(offset=element.xy,
                              layer=(element.layer, element.text_type),
                              string=element.string.decode('ASCII'))
                pat.labels.append(label)

            elif isinstance(element, gdsii.elements.SRef):
                pat.subpatterns.append(_sref_to_subpat(element))

            elif isinstance(element, gdsii.elements.ARef):
                pat.subpatterns.append(_aref_to_gridrep(element))

        patterns.append(pat)

    # Create a dict of {pattern.name: pattern, ...}, then fix up all subpattern.pattern entries
    #  according to the subpattern.ref_name (which is deleted after use).
    patterns_dict = dict(((p.name, p) for p in patterns))
    for p in patterns_dict.values():
        for sp in p.subpatterns:
            sp.pattern = patterns_dict[sp.ref_name.decode('ASCII')]
            del sp.ref_name

    return patterns_dict, library_info


def _mlayer2gds(mlayer):
    if is_scalar(mlayer):
        layer = mlayer
        data_type = 0
    else:
        layer = mlayer[0]
        if len(mlayer) > 1:
            data_type = mlayer[1]
        else:
            data_type = 0
    return layer, data_type


def _sref_to_subpat(element: gdsii.elements.SRef) -> SubPattern:
    # Helper function to create a SubPattern from an SREF. Sets subpat.pattern to None
    #  and sets the instance attribute .ref_name to the struct_name.
    #
    # BUG: "Absolute" means not affected by parent elements.
    #       That's not currently supported by masque at all, so need to either tag it and
    #       undo the parent transformations, or implement it in masque.
    subpat = SubPattern(pattern=None, offset=element.xy)
    subpat.ref_name = element.struct_name
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
            subpat.mirror(axis=0)
    return subpat


def _aref_to_gridrep(element: gdsii.elements.ARef) -> GridRepetition:
    # Helper function to create a GridRepetition from an AREF. Sets gridrep.pattern to None
    #  and sets the instance attribute .ref_name to the struct_name.
    #
    # BUG: "Absolute" means not affected by parent elements.
    #       That's not currently supported by masque at all, so need to either tag it and
    #       undo the parent transformations, or implement it in masque.i

    rotation = 0
    offset = numpy.array(element.xy[0])
    scale = 1
    mirror_signs = numpy.ones(2)

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
            mirror_signs[1] = -1

    counts = [element.cols, element.rows]
    vec_a0 = element.xy[1] - offset
    vec_b0 = element.xy[2] - offset

    a_vector = numpy.dot(rotation_matrix_2d(-rotation), vec_a0 / scale / counts[0]) * mirror_signs
    b_vector = numpy.dot(rotation_matrix_2d(-rotation), vec_b0 / scale / counts[1]) * mirror_signs


    gridrep = GridRepetition(pattern=None,
                            a_vector=a_vector,
                            b_vector=b_vector,
                            a_count=counts[0],
                            b_count=counts[1],
                            offset=offset,
                            rotation=rotation,
                            scale=scale,
                            mirrored=(mirror_signs[::-1] == -1))
    gridrep.ref_name = element.struct_name

    return gridrep


def _subpatterns_to_refs(subpatterns: List[SubPattern or GridRepetition]
                        ) -> List[gdsii.elements.ARef or gdsii.elements.SRef]:
    #  strans must be set for angle and mag to take effect
    refs = []
    for subpat in subpatterns:
        encoded_name = subpat.pattern.name

        if isinstance(subpat, GridRepetition):
            mirror_signs = (-1) ** numpy.array(subpat.mirrored)
            xy = numpy.array(subpat.offset) + [
                  [0, 0],
                  numpy.dot(rotation_matrix_2d(subpat.rotation), subpat.a_vector * mirror_signs) * subpat.scale * subpat.a_count,
                  numpy.dot(rotation_matrix_2d(subpat.rotation), subpat.b_vector * mirror_signs) * subpat.scale * subpat.b_count,
                 ]
            ref = gdsii.elements.ARef(struct_name=encoded_name,
                                       xy=numpy.round(xy).astype(int),
                                       cols=numpy.round(subpat.a_count).astype(int),
                                       rows=numpy.round(subpat.b_count).astype(int))
        else:
            ref = gdsii.elements.SRef(struct_name=encoded_name,
                                      xy=numpy.round([subpat.offset]).astype(int))

        ref.strans = 0
        ref.angle = subpat.rotation * 180 / numpy.pi
        mirror_x, mirror_y = subpat.mirrored
        if mirror_x and mirror_y:
            ref.angle += 180
        elif mirror_x:
            ref.strans = set_bit(ref.strans, 15 - 0, True)
        elif mirror_y:
            ref.angle += 180
            ref.strans = set_bit(ref.strans, 15 - 0, True)
        ref.angle %= 360
        ref.mag = subpat.scale

        refs.append(ref)
    return refs


def _shapes_to_elements(shapes: List[Shape],
                        polygonize_paths: bool = False
                       ) -> List[gdsii.elements.Boundary]:
    elements = []
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


def _disambiguate_pattern_names(patterns):
    used_names = []
    for pat in patterns:
        sanitized_name = re.compile('[^A-Za-z0-9_\?\$]').sub('_', pat.name)

        i = 0
        suffixed_name = sanitized_name
        while suffixed_name in used_names or suffixed_name == '':
            suffix = base64.b64encode(struct.pack('>Q', i), b'$?').decode('ASCII')

            suffixed_name = sanitized_name + '$' + suffix[:-1].lstrip('A')
            i += 1

        if sanitized_name == '':
            logger.warning('Empty pattern name saved as "{}"'.format(suffixed_name))
        elif suffixed_name != sanitized_name:
            logger.warning('Pattern name "{}" appears multiple times; renaming to "{}"'.format(pat.name, suffixed_name))

        encoded_name = suffixed_name.encode('ASCII')
        if len(encoded_name) == 0:
            # Should never happen since zero-length names are replaced
            raise PatternError('Zero-length name after sanitize+encode, originally "{}"'.format(pat.name))
        if len(encoded_name) > 32:
            raise PatternError('Pattern name "{}" length > 32 after encode, originally "{}"'.format(encoded_name, pat.name))

        pat.name = encoded_name
        used_names.append(suffixed_name)

