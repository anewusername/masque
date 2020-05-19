"""
OASIS file format readers and writers

Note that OASIS references follow the same convention as `masque`,
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
import base64
import struct
import logging
import pathlib
import gzip
import numpy
from numpy import pi

import fatamorgana
import fatamorgana.records as fatrec
from fatamorgana.basic import PathExtensionScheme

from .utils import mangle_name, make_dose_table
from .. import Pattern, SubPattern, GridRepetition, PatternError, Label, Shape, subpattern_t
from ..shapes import Polygon, Path, Circle
from ..utils import rotation_matrix_2d, get_bit, set_bit, vector2, is_scalar, layer_t
from ..utils import remove_colinear_vertices, normalize_mirror


logger = logging.getLogger(__name__)


logger.warning('OASIS support is experimental and mostly untested!')


path_cap_map = {
                PathExtensionScheme.Flush: Path.Cap.Flush,
                PathExtensionScheme.HalfWidth: Path.Cap.Square,
                PathExtensionScheme.Arbitrary: Path.Cap.SquareCustom,
               }

#TODO implement properties
#TODO implement more shape types?

def build(patterns: Union[Pattern, List[Pattern]],
          units_per_micron: int,
          layer_map: Dict[str, Union[int, Tuple[int, int]]] = None,
          modify_originals: bool = False,
          disambiguate_func: Callable[[Iterable[Pattern]], None] = None,
          ) -> fatamorgana.OasisLayout:
    """
    Convert a `Pattern` or list of patterns to an OASIS stream, writing patterns
     as OASIS cells, subpatterns as Placement records, and other shapes and labels
     mapped to equivalent record types (Polygon, Path, Circle, Text).
     Other shape types may be converted to polygons if no equivalent
     record type exists (or is not implemented here yet).

     For each shape,
        layer is chosen to be equal to `shape.layer` if it is an int,
            or `shape.layer[0]` if it is a tuple
        datatype is chosen to be `shape.layer[1]` if available,
            otherwise `0`
        If a layer map is provided, layer strings will be converted
            automatically, and layer names will be written to the file.

    If you want pattern polygonized with non-default arguments, just call `pattern.polygonize()`
     prior to calling this function.

    Args:
        patterns: A Pattern or list of patterns to convert.
        units_per_micron: Written into the OASIS file, number of grid steps per micrometer.
            All distances are assumed to be an integer multiple of the grid step, and are stored as such.
        layer_map: Dictionary which translates layer names into layer numbers. If this argument is
            provided, input shapes and labels are allowed to have layer names instead of numbers.
            It is assumed that geometry and text share the same layer names, and each name is
            assigned only to a single layer (not a range).
            If more fine-grained control is needed, manually pre-processing shapes' layer names
            into numbers, omit this argument, and manually generate the required
            `fatamorgana.records.LayerName` entries.
            Default is an empty dict (no names provided).
        modify_originals: If `True`, the original pattern is modified as part of the writing
            process. Otherwise, a copy is made and `deepunlock()`-ed.
            Default `False`.
        disambiguate_func: Function which takes a list of patterns and alters them
            to make their names valid and unique. Default is `disambiguate_pattern_names`.

    Returns:
        `fatamorgana.OasisLayout`
    """
    if isinstance(patterns, Pattern):
        patterns = [patterns]

    if layer_map is None:
        layer_map = {}

    if disambiguate_func is None:
        disambiguate_func = disambiguate_pattern_names

    if not modify_originals:
        patterns = [p.deepunlock() for p in copy.deepcopy(patterns)]

    # Create library
    lib = fatamorgana.OasisLayout(unit=units_per_micron, validation=None)

    if layer_map:
        for name, layer_num in layer_map.items():
            layer, data_type = _mlayer2oas(layer_num)
            lib.layers += [
                    fatrec.LayerName(nstring=name,
                                     layer_interval=(layer, layer),
                                     type_interval=(data_type, data_type),
                                     is_textlayer=tt)
                    for tt in (True, False)]

        def layer2oas(mlayer: layer_t) -> Tuple[int, int]:
            layer_num = layer_map[mlayer] if isinstance(mlayer, str) else mlayer
            return _mlayer2oas(layer_num)
    else:
        layer2oas = _mlayer2oas

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {id(pattern): pattern for pattern in patterns}
    for pattern in patterns:
        for i, p in pattern.referenced_patterns_by_id().items():
            patterns_by_id[i] = p

    disambiguate_func(patterns_by_id.values())

    # Now create a structure for each pattern
    for pat in patterns_by_id.values():
        structure = fatamorgana.Cell(name=pat.name)
        lib.cells.append(structure)

        structure.geometry += _shapes_to_elements(pat.shapes, layer2oas)
        structure.geometry += _labels_to_texts(pat.labels, layer2oas)
        structure.placements += _subpatterns_to_refs(pat.subpatterns)

    return lib


def write(patterns: Union[List[Pattern], Pattern],
          stream: io.BufferedIOBase,
          *args,
          **kwargs):
    """
    Write a `Pattern` or list of patterns to a OASIS file. See `oasis.build()`
      for details.

    Args:
        patterns: A Pattern or list of patterns to write to file.
        stream: Stream to write to.
        *args: passed to `oasis.build()`
        **kwargs: passed to `oasis.build()`
    """
    lib = build(patterns, *args, **kwargs)
    lib.write(stream)


def writefile(patterns: Union[List[Pattern], Pattern],
              filename: Union[str, pathlib.Path],
              *args,
              **kwargs,
              ):
    """
    Wrapper for `oasis.write()` that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.

    Args:
        patterns: `Pattern` or list of patterns to save
        filename: Filename to save to.
        *args: passed to `oasis.write`
        **kwargs: passed to `oasis.write`
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
    Wrapper for `oasis.read()` that takes a filename or path instead of a stream.

    Will automatically decompress files with a .gz suffix.

    Args:
        filename: Filename to save to.
        *args: passed to `oasis.read`
        **kwargs: passed to `oasis.read`
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
         clean_vertices: bool = True,
         ) -> Tuple[Dict[str, Pattern], Dict[str, Any]]:
    """
    Read a OASIS file and translate it into a dict of Pattern objects. OASIS cells are
     translated into Pattern objects; Polygons are translated into polygons, and Placements
     are translated into SubPattern or GridRepetition objects.

    Additional library info is returned in a dict, containing:
      'units_per_micrometer': number of database units per micrometer (all values are in database units)

    Args:
        stream: Stream to read from.
        clean_vertices: If `True`, remove any redundant vertices when loading polygons.
            The cleaning process removes any polygons with zero area or <3 vertices.
            Default `True`.

    Returns:
        - Dict of `pattern_name`:`Pattern`s generated from OASIS cells
        - Dict of OASIS library info
    """

    lib = fatamorgana.OasisLayout.read(stream)

    library_info: Dict[str, Any] = {
            'units_per_micrometer': lib.unit,
            }

    layer_map = {}
    for layer_name in lib.layers:
        layer_map[str(layer_name.nstring)] = layer_name
    library_info['layer_map'] = layer_map

    patterns = []
    for cell in lib.cells:
        if isinstance(cell.name, int):
            cell_name = lib.cellnames[cell.name].string
        else:
            cell_name = cell.name.string

        pat = Pattern(name=cell_name)
        for element in cell.geometry:
            if isinstance(element, fatrec.XElement):
                logger.warning('Skipping XElement record')
                continue

            if element.repetition is not None:
                # note XELEMENT has no repetition
                raise PatternError('masque OASIS reader does not implement repetitions for shapes yet')

            # Switch based on element type:
            if isinstance(element, fatrec.Polygon):
                vertices = numpy.cumsum(numpy.vstack(((0, 0), element.get_point_list())), axis=0)
                poly = Polygon(vertices=vertices,
                               layer=element.get_layer_tuple(),
                               offset=element.get_xy())

                if clean_vertices:
                    try:
                        poly.clean_vertices()
                    except PatternError:
                        continue

                pat.shapes.append(poly)

            elif isinstance(element, fatrec.Path):
                vertices = numpy.cumsum(numpy.vstack(((0, 0), element.get_point_list())), axis=0)

                cap_start = path_cap_map[element.get_extension_start()[0]]
                cap_end   = path_cap_map[element.get_extension_end()[0]]
                if cap_start != cap_end:
                    raise Exception('masque does not support multiple cap types on a single path.')      #TODO handle multiple cap types
                cap = cap_start

                path_args: Dict[str, Any] = {}
                if cap == Path.Cap.SquareCustom:
                    path_args['cap_extensions'] = numpy.array((element.get_extension_start()[1],
                                                               element.get_extension_end()[1]))
                path = Path(vertices=vertices,
                            layer=element.get_layer_tuple(),
                            offset=element.get_xy(),
                            width=element.get_half_width() * 2,
                            cap=cap,
                            **path_args)

                if clean_vertices:
                    try:
                        path.clean_vertices()
                    except PatternError as err:
                        continue

                pat.shapes.append(path)

            elif isinstance(element, fatrec.Rectangle):
                width = element.get_width()
                height = element.get_height()
                rect = Polygon(layer=element.get_layer_tuple(),
                               offset=element.get_xy(),
                               vertices=numpy.array(((0, 0), (1, 0), (1, 1), (0, 1))) * (width, height),
                               )
                pat.shapes.append(rect)

            elif isinstance(element, fatrec.Trapezoid):
                vertices = numpy.array(((0, 0), (1, 0), (1, 1), (0, 1))) * (element.get_width(), element.get_height())
                a = element.get_delta_a()
                b = element.get_delta_b()
                if element.get_is_vertical():
                    if a > 0:
                        vertices[0, 1] += a
                    else:
                        vertices[3, 1] += a

                    if b > 0:
                        vertices[2, 1] -= b
                    else:
                        vertices[1, 1] -= b
                else:
                    if a > 0:
                        vertices[1, 0] += a
                    else:
                        vertices[0, 0] += a

                    if b > 0:
                        vertices[3, 0] -= b
                    else:
                        vertices[2, 0] -= b

                trapz = Polygon(layer=element.get_layer_tuple(),
                                offset=element.get_xy(),
                                vertices=vertices,
                                )
                pat.shapes.append(trapz)

            elif isinstance(element, fatrec.CTrapezoid):
                cttype = element.get_ctrapezoid_type()
                height = element.get_height()
                width = element.get_width()

                vertices = numpy.array(((0, 0), (1, 0), (1, 1), (0, 1))) * (width, height)

                if cttype in (0, 4, 7):
                    vertices[2, 0] -= height
                if cttype in (1, 5, 6):
                    vertices[3, 0] -= height
                if cttype in (2, 4, 6):
                    vertices[1, 0] += height
                if cttype in (3, 5, 7):
                    vertices[0, 0] += height

                if cttype in (8, 12, 15):
                    vertices[2, 0] -= width
                if cttype in (9, 13, 14):
                    vertices[1, 0] -= width
                if cttype in (10, 12, 14):
                    vertices[3, 0] += width
                if cttype in (11, 13, 15):
                    vertices[0, 0] += width

                if cttype == 16:
                    vertices = vertices[[0, 1, 3], :]
                elif cttype == 17:
                    vertices = vertices[[0, 1, 2], :]
                elif cttype == 18:
                    vertices = vertices[[0, 2, 3], :]
                elif cttype == 19:
                    vertices = vertices[[1, 2, 3], :]
                elif cttype == 20:
                    vertices = vertices[[0, 1, 3], :]
                    vertices[1, 0] += height
                elif cttype == 21:
                    vertices = vertices[[0, 1, 2], :]
                    vertices[0, 0] += height
                elif cttype == 22:
                    vertices = vertices[[0, 1, 3], :]
                    vertices[3, 1] += width
                elif cttype == 23:
                    vertices = vertices[[0, 2, 3], :]
                    vertices[0, 1] += width

                ctrapz = Polygon(layer=element.get_layer_tuple(),
                                offset=element.get_xy(),
                                vertices=vertices,
                                )
                pat.shapes.append(ctrapz)

            elif isinstance(element, fatrec.Circle):
                circle = Circle(layer=element.get_layer_tuple(),
                                offset=element.get_xy(),
                                radius=float(element.get_radius()))
                pat.shapes.append(circle)

            elif isinstance(element, fatrec.Text):
                label = Label(layer=element.get_layer_tuple(),
                              offset=element.get_xy(),
                              string=str(element.get_string()))
                pat.labels.append(label)

            else:
                logger.warning(f'Skipping record {element} (unimplemented)')
                continue

        for placement in cell.placements:
            pat.subpatterns += _placement_to_subpats(placement)

        patterns.append(pat)

    # Create a dict of {pattern.name: pattern, ...}, then fix up all subpattern.pattern entries
    #  according to the subpattern.identifier (which is deleted after use).
    patterns_dict = dict(((p.name, p) for p in patterns))
    for p in patterns_dict.values():
        for sp in p.subpatterns:
            ident = sp.identifier[0]
            name = ident if isinstance(ident, str) else lib.cellnames[ident].string
            sp.pattern = patterns_dict[name]
            del sp.identifier

    return patterns_dict, library_info


def _mlayer2oas(mlayer: layer_t) -> Tuple[int, int]:
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
        raise PatternError(f'Invalid layer for OASIS: {layer}. Note that OASIS layers cannot be '
                            'strings unless a layer map is provided.')
    return layer, data_type


def _placement_to_subpats(placement: fatrec.Placement) -> List[subpattern_t]:
    """
    Helper function to create a SubPattern from a placment. Sets subpat.pattern to None
     and sets the instance .identifier to (struct_name,).
    """
    xy = numpy.array((placement.x, placement.y))
    mag = placement.magnification if placement.magnification is not None else 1
    pname = placement.get_name()
    name = pname if isinstance(pname, int) else pname.string
    args: Dict[str, Any] = {
       'pattern': None,
       'mirrored': (placement.flip, False),
       'rotation': float(placement.angle * pi/180),
       'scale': mag,
       'identifier': (name,),
       }

    subpats: List[subpattern_t]
    rep = placement.repetition
    if isinstance(rep, fatamorgana.GridRepetition):
        subpat = GridRepetition(a_vector=rep.a_vector,
                                b_vector=rep.b_vector,
                                a_count=rep.a_count,
                                b_count=rep.b_count,
                                offset=xy,
                                **args)
        subpats = [subpat]
    elif isinstance(rep, fatamorgana.ArbitraryRepetition):
        subpats = []
        for rep_offset in numpy.cumsum(numpy.column_stack((rep.x_displacements,
                                                           rep.y_displacements))):
            subpats.append(SubPattern(offset=xy + rep_offset, **args))
    elif rep is None:
        subpats = [SubPattern(offset=xy, **args)]
    return subpats


def _subpatterns_to_refs(subpatterns: List[subpattern_t]
                        ) -> List[fatrec.Placement]:
    refs = []
    for subpat in subpatterns:
        if subpat.pattern is None:
            continue

        # Note: OASIS mirrors first and rotates second
        mirror_across_x, extra_angle = normalize_mirror(subpat.mirrored)
        xy = numpy.round(subpat.offset).astype(int)
        args: Dict[str, Any] = {
            'x': xy[0],
            'y': xy[1],
            }

        if isinstance(subpat, GridRepetition):
            args['repetition'] = fatamorgana.GridRepetition(
                              a_vector=numpy.round(subpat.a_vector).astype(int),
                              b_vector=numpy.round(subpat.b_vector).astype(int),
                              a_count=numpy.round(subpat.a_count).astype(int),
                              b_count=numpy.round(subpat.b_count).astype(int))

        angle = ((subpat.rotation + extra_angle) * 180 / numpy.pi) % 360
        ref = fatrec.Placement(
                name=subpat.pattern.name,
                flip=mirror_across_x,
                angle=angle,
                magnification=subpat.scale,
                **args)

        refs.append(ref)
    return refs


def _shapes_to_elements(shapes: List[Shape],
                        layer2oas: Callable[[layer_t], Tuple[int, int]],
                       ) -> List[Union[fatrec.Polygon, fatrec.Path, fatrec.Circle]]:
    # Add a Polygon record for each shape, and Path elements if necessary
    elements: List[Union[fatrec.Polygon, fatrec.Path, fatrec.Circle]] = []
    for shape in shapes:
        layer, datatype = layer2oas(shape.layer)
        if isinstance(shape, Circle):
            offset = numpy.round(shape.offset).astype(int)
            radius = numpy.round(shape.radius).astype(int)
            circle = fatrec.Circle(layer=layer,
                                   datatype=datatype,
                                   radius=radius,
                                   x=offset[0],
                                   y=offset[1])
            elements.append(circle)
        elif isinstance(shape, Path):
            xy = numpy.round(shape.offset + shape.vertices[0]).astype(int)
            deltas = numpy.round(numpy.diff(shape.vertices, axis=0)).astype(int)
            half_width = numpy.round(shape.width / 2).astype(int)
            path_type = next(k for k, v in path_cap_map.items() if v == shape.cap)    #reverse lookup
            extension_start = (path_type, shape.cap_extensions[0] if shape.cap_extensions is not None else None)
            extension_end = (path_type, shape.cap_extensions[1] if shape.cap_extensions is not None else None)
            path = fatrec.Path(layer=layer,
                               datatype=datatype,
                               point_list=deltas,
                               half_width=half_width,
                               x=xy[0],
                               y=xy[1],
                               extension_start=extension_start,       #TODO implement multiple cap types?
                               extension_end=extension_end,
                               )
            elements.append(path)
        else:
            for polygon in shape.to_polygons():
                xy = numpy.round(polygon.offset + polygon.vertices[0]).astype(int)
                points = numpy.round(numpy.diff(polygon.vertices, axis=0)).astype(int)
                elements.append(fatrec.Polygon(layer=layer,
                                               datatype=datatype,
                                               x=xy[0],
                                               y=xy[1],
                                               point_list=points))
    return elements


def _labels_to_texts(labels: List[Label],
                     layer2oas: Callable[[layer_t], Tuple[int, int]],
                     ) -> List[fatrec.Text]:
    texts = []
    for label in labels:
        layer, datatype = layer2oas(label.layer)
        xy = numpy.round(label.offset).astype(int)
        texts.append(fatrec.Text(layer=layer,
                                 datatype=datatype,
                                 x=xy[0],
                                 y=xy[1],
                                 string=label.string))
    return texts


def disambiguate_pattern_names(patterns,
                               dup_warn_filter: Callable[[str,], bool] = None,      # If returns False, don't warn about this name
                               ):
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
            if dup_warn_filter is None or dup_warn_filter(pat.name):
                logger.warning('Pattern name "{}" ({}) appears multiple times;\n renaming to "{}"'.format(
                                pat.name, sanitized_name, suffixed_name))

        if len(suffixed_name) == 0:
            # Should never happen since zero-length names are replaced
            raise PatternError('Zero-length name after sanitize+encode,\n originally "{}"'.format(pat.name))

        pat.name = suffixed_name
        used_names.append(suffixed_name)
