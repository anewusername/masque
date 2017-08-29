"""
GDSII file format readers and writers
"""

import gdsii.library
import gdsii.structure
import gdsii.elements

from typing import List, Any, Dict
import re
import numpy

from .utils import mangle_name, make_dose_table
from .. import Pattern, SubPattern, PatternError
from ..shapes import Polygon
from ..utils import rotation_matrix_2d, get_bit, vector2


__author__ = 'Jan Petykiewicz'


def write(pattern: Pattern,
          filename: str,
          meters_per_unit: float):
    """
    Write a Pattern to a GDSII file, by first calling .polygonize() on it
     to change the shapes into polygons, and then writing patterns as GDSII
     structures, polygons as boundary elements, and subpatterns as structure
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

    :param pattern: A Pattern to write to file. Modified by this function.
    :param filename: Filename to write to.
    :param meters_per_unit: Written into the GDSII file, meters per length unit.
    """
    # Create library
    lib = gdsii.library.Library(version=600,
                                name='masque-write_dose2dtype'.encode('ASCII'),
                                logical_unit=1,
                                physical_unit=meters_per_unit)

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {**(pattern.referenced_patterns_by_id()), id(pattern): pattern}

    # Now create a structure for each pattern, and add in any Boundary and SREF elements
    for pat in patterns_by_id.values():
        sanitized_name = re.compile('[^A-Za-z0-9_\?\$]').sub('_', pattern.name)
        structure = gdsii.structure.Structure(name=sanitized_name.encode('ASCII'))
        lib.append(structure)

        # Add a Boundary element for each shape
        for shape in pat.shapes:
            for polygon in shape.to_polygons():
                xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                if hasattr(polygon.layer, '__len__'):
                    layer = polygon.layer[0]
                    if len(polygon.layer) > 1:
                        data_type = polygon.layer[1]
                    else:
                        data_type = 0
                else:
                    layer = polygon.layer
                    data_type = 0
                structure.append(gdsii.elements.Boundary(layer=layer,
                                                         data_type=data_type,
                                                         xy=xy_closed))
        # Add an SREF for each subpattern entry
        #  strans must be set for angle and mag to take effect
        for subpat in pat.subpatterns:
            sanitized_name = re.compile('[^A-Za-z0-9_\?\$]').sub('_', subpat.name)
            sref = gdsii.elements.SRef(struct_name=sanitized_name.encode('ASCII'),
                                       xy=numpy.round([subpat.offset]).astype(int))
            sref.strans = 0
            sref.angle = subpat.rotation
            sref.mag = subpat.scale
            structure.append(sref)

    with open(filename, mode='wb') as stream:
        lib.save(stream)


def write_dose2dtype(pattern: Pattern,
                     filename: str,
                     meters_per_unit: float) -> List[float]:
    """
    Write a Pattern to a GDSII file, by first calling .polygonize() on it
     to change the shapes into polygons, and then writing patterns as GDSII
     structures, polygons as boundary elements, and subpatterns as structure
     references (sref).

     For each shape,
        layer is chosen to be equal to shape.layer if it is an int,
            or shape.layer[0] if it is a tuple
        datatype is chosen arbitrarily, based on calcualted dose for each shape.
            Shapes with equal calcualted dose will have the same datatype.
            A list of doses is retured, providing a mapping between datatype
            (list index) and dose (list entry).

    Note that this function modifies the Pattern.

    It is often a good idea to run pattern.subpatternize() prior to calling this function,
     especially if calling .polygonize() will result in very many vertices.

    If you want pattern polygonized with non-default arguments, just call pattern.polygonize()
     prior to calling this function.

    :param pattern: A Pattern to write to file. Modified by this function.
    :param filename: Filename to write to.
    :param meters_per_unit: Written into the GDSII file, meters per length unit.
    :returns: A list of doses, providing a mapping between datatype (int, list index)
                and dose (float, list entry).
    """
    # Create library
    lib = gdsii.library.Library(version=600,
                                name='masque-write_dose2dtype'.encode('ASCII'),
                                logical_unit=1,
                                physical_unit=meters_per_unit)

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {**(pattern.referenced_patterns_by_id()), id(pattern): pattern}

    # Get a table of (id(subpat.pattern), written_dose) for each subpattern
    sd_table = make_dose_table(pattern)

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

        structure = gdsii.structure.Structure(name=mangle_name(pat, pat_dose).encode('ASCII'))
        lib.append(structure)

        # Add a Boundary element for each shape
        for shape in pat.shapes:
            for polygon in shape.to_polygons():
                data_type = dose_vals_list.index(polygon.dose * pat_dose)
                xy_open = numpy.round(polygon.vertices + polygon.offset).astype(int)
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                if hasattr(polygon.layer, '__len__'):
                    layer = polygon.layer[0]
                else:
                    layer = polygon.layer
                structure.append(gdsii.elements.Boundary(layer=layer,
                                                         data_type=data_type,
                                                         xy=xy_closed))
        # Add an SREF for each subpattern entry
        #  strans must be set for angle and mag to take effect
        for subpat in pat.subpatterns:
            dose_mult = subpat.dose * pat_dose
            sref = gdsii.elements.SRef(struct_name=mangle_name(subpat.pattern,  dose_mult).encode('ASCII'),
                                       xy=numpy.round([subpat.offset]).astype(int))
            sref.strans = 0
            sref.angle = subpat.rotation
            sref.mag = subpat.scale
            structure.append(sref)

    with open(filename, mode='wb') as stream:
        lib.save(stream)

    return dose_vals_list


def read_dtype2dose(filename: str) -> (List[Pattern], Dict[str, Any]):
    """
    Alias for read(filename, use_dtype_as_dose=True)
    """
    return read(filename, use_dtype_as_dose=True)


def read(filename: str, use_dtype_as_dose=False) -> (List[Pattern], Dict[str, Any]):
    """
    Read a gdsii file and translate it into a list of Pattern objects. GDSII structures are
     translated into Pattern objects; boundaries are translated into polygons, and srefs and arefs
     are translated into SubPattern objects.

    :param filename: Filename specifying a GDSII file to read from.
    :param use_dtype_as_dose: If false, set each polygon's layer to (gds_layer, gds_datatype).
            If true, set the layer to gds_layer and the dose to gds_datatype.
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
        # BUG: Figure out what "absolute" means in the context of elements and if the current
        #       behavior is correct
        # BUG: Need to check STRANS bit 0 to handle x-reflection
        subpat = SubPattern(pattern=None, offset=offset)
        subpat.ref_name = element.struct_name
        if element.strans is not None:
            if element.mag is not None:
                subpat.scale = element.mag
                # Bit 13 means absolute scale
                if get_bit(element.strans, 13):
                    subpat.offset *= subpat.scale
            if element.angle is not None:
                subpat.rotation = element.angle
                # Bit 14 means absolute rotation
                if get_bit(element.strans, 14):
                    subpat.offset = numpy.dot(rotation_matrix_2d(subpat.rotation), subpat.offset)
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
                pat.shapes.append(shape)

            elif isinstance(element, gdsii.elements.SRef):
                pat.subpatterns.append(ref_element_to_subpat(element, element.xy))

            elif isinstance(element, gdsii.elements.ARef):
                for offset in element.xy:
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
