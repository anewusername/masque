"""
GDSII file format readers and writers
"""
# python-gdsii
import gdsii.library
import gdsii.structure
import gdsii.elements

from typing import List, Any, Dict
import re
import numpy

from .utils import mangle_name, make_dose_table
from .. import Pattern, SubPattern, PatternError
from ..shapes import Polygon
from ..utils import rotation_matrix_2d, get_bit, set_bit, vector2, is_scalar


__author__ = 'Jan Petykiewicz'


def write(patterns: Pattern or List[Pattern],
          filename: str,
          meters_per_unit: float,
          logical_units_per_unit: float = 1):
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

    Note that this function modifies the Pattern.

    It is often a good idea to run pattern.subpatternize() prior to calling this function,
     especially if calling .polygonize() will result in very many vertices.

    If you want pattern polygonized with non-default arguments, just call pattern.polygonize()
     prior to calling this function.

    :param patterns: A Pattern or list of patterns to write to file. Modified by this function.
    :param filename: Filename to write to.
    :param meters_per_unit: Written into the GDSII file, meters per (database) length unit.
        All distances are assumed to be an integer multiple of this unit, and are stored as such.
    :param logical_units_per_unit: Written into the GDSII file. Allows the GDSII to specify a
        "logical" unit which is different from the "database" unit, for display purposes.
        Default 1.
    """
    # Create library
    lib = gdsii.library.Library(version=600,
                                name='masque-write_dose2dtype'.encode('ASCII'),
                                logical_unit=logical_units_per_unit,
                                physical_unit=meters_per_unit)

    if isinstance(patterns, Pattern):
        patterns = [patterns]

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {id(pattern): pattern for pattern in patterns}
    for pattern in patterns:
        patterns_by_id.update(pattern.referenced_patterns_by_id())

    # Now create a structure for each pattern, and add in any Boundary and SREF elements
    for pat in patterns_by_id.values():
        sanitized_name = re.compile('[^A-Za-z0-9_\?\$]').sub('_', pat.name)
        encoded_name = sanitized_name.encode('ASCII')
        if len(encoded_name) == 0:
            raise PatternError('Zero-length name after sanitize+encode, originally "{}"'.format(pat.name))
        structure = gdsii.structure.Structure(name=encoded_name)
        lib.append(structure)

        # Add a Boundary element for each shape
        for shape in pat.shapes:
            for polygon in shape.to_polygons():
                xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                if is_scalar(polygon.layer):
                    layer = polygon.layer
                    data_type = 0
                else:
                    layer = polygon.layer[0]
                    if len(polygon.layer) > 1:
                        data_type = polygon.layer[1]
                    else:
                        data_type = 0
                structure.append(gdsii.elements.Boundary(layer=layer,
                                                         data_type=data_type,
                                                         xy=xy_closed))
        # Add an SREF for each subpattern entry
        #  strans must be set for angle and mag to take effect
        for subpat in pat.subpatterns:
            sanitized_name = re.compile('[^A-Za-z0-9_\?\$]').sub('_', subpat.pattern.name)
            encoded_name = sanitized_name.encode('ASCII')
            if len(encoded_name) == 0:
                raise PatternError('Zero-length name after sanitize+encode, originally "{}"'.format(subpat.pattern.name))
            sref = gdsii.elements.SRef(struct_name=encoded_name,
                                       xy=numpy.round([subpat.offset]).astype(int))
            sref.strans = 0
            sref.angle = subpat.rotation * 180 / numpy.pi
            mirror_x, mirror_y = subpat.mirrored
            if mirror_y and mirror_y:
                sref.angle += 180
            elif mirror_x:
                sref.strans = set_bit(sref.strans, 15 - 0, True)
            elif mirror_y:
                sref.angle += 180
                sref.strans = set_bit(sref.strans, 15 - 0, True)
            sref.mag = subpat.scale
            structure.append(sref)

    with open(filename, mode='wb') as stream:
        lib.save(stream)


def write_dose2dtype(patterns: Pattern or List[Pattern],
                     filename: str,
                     meters_per_unit: float,
                     logical_units_per_unit: float = 1
                     ) -> List[float]:
    """
    Write a Pattern or list of patterns to a GDSII file, by first calling
     .polygonize() to change the shapes into polygons, and then writing patterns
     as GDSII structures, polygons as boundary elements, and subpatterns as structure
     references (sref).

     For each shape,
        layer is chosen to be equal to shape.layer if it is an int,
            or shape.layer[0] if it is a tuple
        datatype is chosen arbitrarily, based on calcualted dose for each shape.
            Shapes with equal calcualted dose will have the same datatype.
            A list of doses is retured, providing a mapping between datatype
            (list index) and dose (list entry).

    Note that this function modifies the Pattern(s).

    It is often a good idea to run pattern.subpatternize() prior to calling this function,
     especially if calling .polygonize() will result in very many vertices.

    If you want pattern polygonized with non-default arguments, just call pattern.polygonize()
     prior to calling this function.

    :param patterns: A Pattern or list of patterns to write to file. Modified by this function.
    :param filename: Filename to write to.
    :param meters_per_unit: Written into the GDSII file, meters per (database) length unit.
        All distances are assumed to be an integer multiple of this unit, and are stored as such.
    :param logical_units_per_unit: Written into the GDSII file. Allows the GDSII to specify a
        "logical" unit which is different from the "database" unit, for display purposes.
        Default 1.
    :returns: A list of doses, providing a mapping between datatype (int, list index)
                and dose (float, list entry).
    """
    # Create library
    lib = gdsii.library.Library(version=600,
                                name='masque-write_dose2dtype'.encode('ASCII'),
                                logical_unit=logical_units_per_unit,
                                physical_unit=meters_per_unit)

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

    dose_vals_list = list(dose_vals)

    # Now create a structure for each row in sd_table (ie, each pattern + dose combination)
    #  and add in any Boundary and SREF elements
    for pat_id, pat_dose in sd_table:
        pat = patterns_by_id[pat_id]

        encoded_name = mangle_name(pat, pat_dose).encode('ASCII')
        if len(encoded_name) == 0:
            raise PatternError('Zero-length name after mangle+encode, originally "{}"'.format(pat.name))
        structure = gdsii.structure.Structure(name=encoded_name)
        lib.append(structure)

        # Add a Boundary element for each shape
        for shape in pat.shapes:
            for polygon in shape.to_polygons():
                data_type = dose_vals_list.index(polygon.dose * pat_dose)
                xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                if is_scalar(polygon.layer):
                    layer = polygon.layer
                else:
                    layer = polygon.layer[0]
                structure.append(gdsii.elements.Boundary(layer=layer,
                                                         data_type=data_type,
                                                         xy=xy_closed))
        # Add an SREF for each subpattern entry
        #  strans must be set for angle and mag to take effect
        for subpat in pat.subpatterns:
            dose_mult = subpat.dose * pat_dose
            encoded_name = mangle_name(subpat.pattern, dose_mult).encode('ASCII')
            if len(encoded_name) == 0:
                raise PatternError('Zero-length name after mangle+encode, originally "{}"'.format(subpat.pattern.name))
            sref = gdsii.elements.SRef(struct_name=encoded_name,
                                       xy=numpy.round([subpat.offset]).astype(int))
            sref.strans = 0
            sref.angle = subpat.rotation * 180 / numpy.pi
            sref.mag = subpat.scale
            mirror_x, mirror_y = subpat.mirrored
            if mirror_y and mirror_y:
                sref.angle += 180
            elif mirror_x:
                sref.strans = set_bit(sref.strans, 15 - 0, True)
            elif mirror_y:
                sref.angle += 180
                sref.strans = set_bit(sref.strans, 15 - 0, True)
            structure.append(sref)

    with open(filename, mode='wb') as stream:
        lib.save(stream)

    return dose_vals_list


def read_dtype2dose(filename: str) -> (List[Pattern], Dict[str, Any]):
    """
    Alias for read(filename, use_dtype_as_dose=True)
    """
    return read(filename, use_dtype_as_dose=True)


def read(filename: str,
         use_dtype_as_dose: bool = False,
         clean_vertices: bool = True,
         ) -> (List[Pattern], Dict[str, Any]):
    """
    Read a gdsii file and translate it into a list of Pattern objects. GDSII structures are
     translated into Pattern objects; boundaries are translated into polygons, and srefs and arefs
     are translated into SubPattern objects.

    :param filename: Filename specifying a GDSII file to read from.
    :param use_dtype_as_dose: If false, set each polygon's layer to (gds_layer, gds_datatype).
            If true, set the layer to gds_layer and the dose to gds_datatype.
            Default False.
    :param clean_vertices: If true, remove any redundant vertices when loading polygons.
            The cleaning process removes any polygons with zero area or <3 vertices.
            Default True.
    :return: Tuple: (List of Patterns generated GDSII structures, Dict of GDSII library info)
    """

    with open(filename, mode='rb') as stream:
        lib = gdsii.library.Library.load(stream)

    library_info = {'name': lib.name.decode('ASCII'),
                    'physical_unit': lib.physical_unit,
                    'logical_unit': lib.logical_unit,
                    }

    def ref_element_to_subpat(element, offset: vector2) -> SubPattern:
        # Helper function to create a SubPattern from an SREF or AREF. Sets subpat.pattern to None
        #  and sets the instance attribute .ref_name to the struct_name.
        #
        # BUG: "Absolute" means not affected by parent elements.
        #       That's not currently supported by masque at all, so need to either tag it and
        #       undo the parent transformations, or implement it in masque.
        subpat = SubPattern(pattern=None, offset=offset)
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


    patterns = []
    for structure in lib:
        pat = Pattern(name=structure.name.decode('ASCII'))
        for element in structure:
            # Switch based on element type:
            if isinstance(element, gdsii.elements.Boundary):
                if use_dtype_as_dose:
                    shape = Polygon(vertices=element.xy[:-1],
                                    dose=element.data_type,
                                    layer=element.layer)
                else:
                    shape = Polygon(vertices=element.xy[:-1],
                                    layer=(element.layer, element.data_type))
                if clean_vertices:
                    try:
                        shape.clean_vertices()
                    except PatternError:
                        continue

                pat.shapes.append(shape)

            elif isinstance(element, gdsii.elements.SRef):
                pat.subpatterns.append(ref_element_to_subpat(element, element.xy))

            elif isinstance(element, gdsii.elements.ARef):
                xy = numpy.array(element.xy)
                origin = xy[0]
                col_spacing = (xy[1] - origin) / element.cols
                row_spacing = (xy[2] - origin) / element.rows

                print(element.xy)
                for c in range(element.cols):
                    for r in range(element.rows):
                        offset = origin + c * col_spacing + r * row_spacing
                        pat.subpatterns.append(ref_element_to_subpat(element, offset))

        patterns.append(pat)

    # Create a dict of {pattern.name: pattern, ...}, then fix up all subpattern.pattern entries
    #  according to the subpattern.ref_name (which is deleted after use).
    patterns_dict = dict(((p.name, p) for p in patterns))
    for p in patterns_dict.values():
        for sp in p.subpatterns:
            sp.pattern = patterns_dict[sp.ref_name.decode('ASCII')]
            del sp.ref_name

    return patterns_dict, library_info
