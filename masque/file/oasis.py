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

Notes:
 * Gzip modification time is set to 0 (start of current epoch, usually 1970-01-01)
"""
from typing import Any, Callable, Iterable, IO, Mapping, cast, Sequence
import logging
import pathlib
import gzip
import string
from pprint import pformat

import numpy
from numpy.typing import ArrayLike, NDArray
import fatamorgana
import fatamorgana.records as fatrec
from fatamorgana.basic import PathExtensionScheme, AString, NString, PropStringReference

from .utils import is_gzipped, tmpfile
from .. import Pattern, Ref, PatternError, LibraryError, Label, Shape
from ..library import Library, ILibrary
from ..shapes import Path, Circle
from ..repetition import Grid, Arbitrary, Repetition
from ..utils import layer_t, annotations_t


logger = logging.getLogger(__name__)


logger.warning('OASIS support is experimental!')


path_cap_map = {
    PathExtensionScheme.Flush: Path.Cap.Flush,
    PathExtensionScheme.HalfWidth: Path.Cap.Square,
    PathExtensionScheme.Arbitrary: Path.Cap.SquareCustom,
    }

#TODO implement more shape types in OASIS?

def rint_cast(val: ArrayLike) -> NDArray[numpy.int64]:
    return numpy.rint(val).astype(numpy.int64)


def build(
        library: Mapping[str, Pattern],           # NOTE: Pattern here should be treated as immutable!
        units_per_micron: int,
        layer_map: dict[str, int | tuple[int, int]] | None = None,
        *,
        annotations: annotations_t | None = None,
        ) -> fatamorgana.OasisLayout:
    """
    Convert a collection of {name: Pattern} pairs to an OASIS stream, writing patterns
     as OASIS cells, refs as Placement records, and mapping other shapes and labels
     to equivalent record types (Polygon, Path, Circle, Text).
     Other shape types may be converted to polygons if no equivalent
     record type exists (or is not implemented here yet).

     For each shape,
        layer is chosen to be equal to `shape.layer` if it is an int,
            or `shape.layer[0]` if it is a tuple
        datatype is chosen to be `shape.layer[1]` if available,
            otherwise `0`
        If a layer map is provided, layer strings will be converted
            automatically, and layer names will be written to the file.

    Other functions you may want to call:
        - `masque.file.oasis.check_valid_names(library.keys())` to check for invalid names
        - `library.dangling_refs()` to check for references to missing patterns
        - `pattern.polygonize()` for any patterns with shapes other
            than `masque.shapes.Polygon`, `masque.shapes.Path`, or `masque.shapes.Circle`

    Args:
        library: A {name: Pattern} mapping of patterns to write.
        units_per_micron: Written into the OASIS file, number of grid steps per micrometer.
            All distances are assumed to be an integer multiple of the grid step, and are stored as such.
        layer_map: dictionary which translates layer names into layer numbers. If this argument is
            provided, input shapes and labels are allowed to have layer names instead of numbers.
            It is assumed that geometry and text share the same layer names, and each name is
            assigned only to a single layer (not a range).
            If more fine-grained control is needed, manually pre-processing shapes' layer names
            into numbers, omit this argument, and manually generate the required
            `fatamorgana.records.LayerName` entries.
            Default is an empty dict (no names provided).
        annotations: dictionary of key-value pairs which are saved as library-level properties

    Returns:
        `fatamorgana.OasisLayout`
    """
    if not isinstance(library, ILibrary):
        if isinstance(library, dict):
            library = Library(library)
        else:
            library = Library(dict(library))

    if layer_map is None:
        layer_map = {}

    if annotations is None:
        annotations = {}

    # Create library
    lib = fatamorgana.OasisLayout(unit=units_per_micron, validation=None)
    lib.properties = annotations_to_properties(annotations)

    if layer_map:
        for name, layer_num in layer_map.items():
            layer, data_type = _mlayer2oas(layer_num)
            lib.layers += [
                fatrec.LayerName(
                    nstring=name,
                    layer_interval=(layer, layer),
                    type_interval=(data_type, data_type),
                    is_textlayer=tt,
                    )
                for tt in (True, False)]

        def layer2oas(mlayer: layer_t) -> tuple[int, int]:
            assert layer_map is not None
            layer_num = layer_map[mlayer] if isinstance(mlayer, str) else mlayer
            return _mlayer2oas(layer_num)
    else:
        layer2oas = _mlayer2oas

    # Now create a structure for each pattern
    for name, pat in library.items():
        structure = fatamorgana.Cell(name=name)
        lib.cells.append(structure)

        structure.properties += annotations_to_properties(pat.annotations)

        structure.geometry += _shapes_to_elements(pat.shapes, layer2oas)
        structure.geometry += _labels_to_texts(pat.labels, layer2oas)
        structure.placements += _refs_to_placements(pat.refs)

    return lib


def write(
        library: Mapping[str, Pattern],           # NOTE: Pattern here should be treated as immutable!
        stream: IO[bytes],
        *args,
        **kwargs,
        ) -> None:
    """
    Write a `Pattern` or list of patterns to a OASIS file. See `oasis.build()`
      for details.

    Args:
        library: A {name: Pattern} mapping of patterns to write.
        stream: Stream to write to.
        *args: passed to `oasis.build()`
        **kwargs: passed to `oasis.build()`
    """
    lib = build(library, *args, **kwargs)
    lib.write(stream)


def writefile(
        library: Mapping[str, Pattern],           # NOTE: Pattern here should be treated as immutable!
        filename: str | pathlib.Path,
        *args,
        **kwargs,
        ) -> None:
    """
    Wrapper for `oasis.write()` that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.

    Args:
        library: A {name: Pattern} mapping of patterns to write.
        filename: Filename to save to.
        *args: passed to `oasis.write`
        **kwargs: passed to `oasis.write`
    """
    path = pathlib.Path(filename)

    with tmpfile(path) as base_stream:
        streams: tuple[Any, ...] = (base_stream,)
        if path.suffix == '.gz':
            stream = cast(IO[bytes], gzip.GzipFile(filename='', mtime=0, fileobj=base_stream, mode='wb'))
            streams += (stream,)
        else:
            stream = base_stream

        try:
            write(library, stream, *args, **kwargs)
        finally:
            for ss in streams:
                ss.close()


def readfile(
        filename: str | pathlib.Path,
        *args,
        **kwargs,
        ) -> tuple[Library, dict[str, Any]]:
    """
    Wrapper for `oasis.read()` that takes a filename or path instead of a stream.

    Will automatically decompress gzipped files.

    Args:
        filename: Filename to save to.
        *args: passed to `oasis.read`
        **kwargs: passed to `oasis.read`
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
        stream: IO[bytes],
        ) -> tuple[Library, dict[str, Any]]:
    """
    Read a OASIS file and translate it into a dict of Pattern objects. OASIS cells are
     translated into Pattern objects; Polygons are translated into polygons, and Placements
     are translated into Ref objects.

    Additional library info is returned in a dict, containing:
      'units_per_micrometer': number of database units per micrometer (all values are in database units)
      'layer_map': Mapping from layer names to fatamorgana.LayerName objects
      'annotations': Mapping of {key: value} pairs from library's properties

    Args:
        stream: Stream to read from.

    Returns:
        - dict of `pattern_name`:`Pattern`s generated from OASIS cells
        - dict of OASIS library info
    """

    lib = fatamorgana.OasisLayout.read(stream)

    library_info: dict[str, Any] = {
        'units_per_micrometer': lib.unit,
        'annotations': properties_to_annotations(lib.properties, lib.propnames, lib.propstrings),
        }

    layer_map = {}
    for layer_name in lib.layers:
        layer_map[str(layer_name.nstring)] = layer_name
    library_info['layer_map'] = layer_map

    mlib = Library()
    for cell in lib.cells:
        if isinstance(cell.name, int):
            cell_name = lib.cellnames[cell.name].nstring.string
        else:
            cell_name = cell.name.string

        pat = Pattern()
        for element in cell.geometry:
            if isinstance(element, fatrec.XElement):
                logger.warning('Skipping XElement record')
                # note XELEMENT has no repetition
                continue

            assert not isinstance(element.repetition, fatamorgana.ReuseRepetition)
            repetition = repetition_fata2masq(element.repetition)

            # Switch based on element type:
            if isinstance(element, fatrec.Polygon):
                # Drop last point (`fatamorgana` returns explicity closed list; we use implicit close)
                # also need `cumsum` to convert from deltas to locations
                vertices = numpy.cumsum(numpy.vstack(((0, 0), element.get_point_list()[:-1])), axis=0)

                annotations = properties_to_annotations(element.properties, lib.propnames, lib.propstrings)
                pat.polygon(
                    vertices=vertices,
                    layer=element.get_layer_tuple(),
                    offset=element.get_xy(),
                    annotations=annotations,
                    repetition=repetition,
                    )
            elif isinstance(element, fatrec.Path):
                vertices = numpy.cumsum(numpy.vstack(((0, 0), element.get_point_list())), axis=0)

                cap_start = path_cap_map[element.get_extension_start()[0]]
                cap_end   = path_cap_map[element.get_extension_end()[0]]
                if cap_start != cap_end:
                    raise Exception('masque does not support multiple cap types on a single path.')      # TODO handle multiple cap types
                cap = cap_start

                path_args: dict[str, Any] = {}
                if cap == Path.Cap.SquareCustom:
                    path_args['cap_extensions'] = numpy.array((
                        element.get_extension_start()[1],
                        element.get_extension_end()[1],
                        ))

                annotations = properties_to_annotations(element.properties, lib.propnames, lib.propstrings)
                pat.path(
                    vertices=vertices,
                    layer=element.get_layer_tuple(),
                    offset=element.get_xy(),
                    repetition=repetition,
                    annotations=annotations,
                    width=element.get_half_width() * 2,
                    cap=cap,
                    **path_args,
                    )

            elif isinstance(element, fatrec.Rectangle):
                width = element.get_width()
                height = element.get_height()
                annotations = properties_to_annotations(element.properties, lib.propnames, lib.propstrings)
                pat.polygon(
                    layer=element.get_layer_tuple(),
                    offset=element.get_xy(),
                    repetition=repetition,
                    vertices=numpy.array(((0, 0), (1, 0), (1, 1), (0, 1))) * (width, height),
                    annotations=annotations,
                    )

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

                annotations = properties_to_annotations(element.properties, lib.propnames, lib.propstrings)
                pat.polygon(
                    layer=element.get_layer_tuple(),
                    offset=element.get_xy(),
                    repetition=repetition,
                    vertices=vertices,
                    annotations=annotations,
                    )

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

                annotations = properties_to_annotations(element.properties, lib.propnames, lib.propstrings)
                pat.polygon(
                    layer=element.get_layer_tuple(),
                    offset=element.get_xy(),
                    repetition=repetition,
                    vertices=vertices,
                    annotations=annotations,
                    )

            elif isinstance(element, fatrec.Circle):
                annotations = properties_to_annotations(element.properties, lib.propnames, lib.propstrings)
                layer = element.get_layer_tuple()
                circle = Circle(
                    offset=element.get_xy(),
                    repetition=repetition,
                    annotations=annotations,
                    radius=float(element.get_radius()),
                    )
                pat.shapes[layer].append(circle)

            elif isinstance(element, fatrec.Text):
                annotations = properties_to_annotations(element.properties, lib.propnames, lib.propstrings)
                str_or_ref = element.get_string()
                if isinstance(str_or_ref, int):
                    string = lib.textstrings[str_or_ref].string
                else:
                    string = str_or_ref.string
                pat.label(
                    layer=element.get_layer_tuple(),
                    offset=element.get_xy(),
                    repetition=repetition,
                    annotations=annotations,
                    string=string,
                    )

            else:
                logger.warning(f'Skipping record {element} (unimplemented)')
                continue

        for placement in cell.placements:
            target, ref = _placement_to_ref(placement, lib)
            pat.refs[target].append(ref)

        mlib[cell_name] = pat

    return mlib, library_info


def _mlayer2oas(mlayer: layer_t) -> tuple[int, int]:
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
        raise PatternError(f'Invalid layer for OASIS: {mlayer}. Note that OASIS layers cannot be '
                           f'strings unless a layer map is provided.')
    return layer, data_type


def _placement_to_ref(placement: fatrec.Placement, lib: fatamorgana.OasisLayout) -> tuple[int | str, Ref]:
    """
    Helper function to create a Ref from a placment. Also returns the placement name (or id).
    """
    assert not isinstance(placement.repetition, fatamorgana.ReuseRepetition)
    xy = numpy.array((placement.x, placement.y))
    mag = placement.magnification if placement.magnification is not None else 1

    pname = placement.get_name()
    name: int | str = pname if isinstance(pname, int) else pname.string       # TODO deal with referenced names

    annotations = properties_to_annotations(placement.properties, lib.propnames, lib.propstrings)
    if placement.angle is None:
        rotation = 0
    else:
        rotation = numpy.deg2rad(float(placement.angle))
    ref = Ref(
        offset=xy,
        mirrored=placement.flip,
        rotation=rotation,
        scale=float(mag),
        repetition=repetition_fata2masq(placement.repetition),
        annotations=annotations,
        )
    return name, ref


def _refs_to_placements(
        refs: dict[str | None, list[Ref]],
        ) -> list[fatrec.Placement]:
    placements = []
    for target, rseq in refs.items():
        if target is None:
            continue
        for ref in rseq:
            # Note: OASIS also mirrors first and rotates second
            frep, rep_offset = repetition_masq2fata(ref.repetition)

            offset = rint_cast(ref.offset + rep_offset)
            angle = numpy.rad2deg(ref.rotation) % 360
            placement = fatrec.Placement(
                name=target,
                flip=ref.mirrored,
                angle=angle,
                magnification=ref.scale,
                properties=annotations_to_properties(ref.annotations),
                x=offset[0],
                y=offset[1],
                repetition=frep,
                )

            placements.append(placement)
    return placements


def _shapes_to_elements(
        shapes: dict[layer_t, list[Shape]],
        layer2oas: Callable[[layer_t], tuple[int, int]],
        ) -> list[fatrec.Polygon | fatrec.Path | fatrec.Circle]:
    # Add a Polygon record for each shape, and Path elements if necessary
    elements: list[fatrec.Polygon | fatrec.Path | fatrec.Circle] = []
    for mlayer, sseq in shapes.items():
        layer, datatype = layer2oas(mlayer)
        for shape in sseq:
            repetition, rep_offset = repetition_masq2fata(shape.repetition)
            properties = annotations_to_properties(shape.annotations)
            if isinstance(shape, Circle):
                offset = rint_cast(shape.offset + rep_offset)
                radius = rint_cast(shape.radius)
                circle = fatrec.Circle(
                    layer=layer,
                    datatype=datatype,
                    radius=cast(int, radius),
                    x=offset[0],
                    y=offset[1],
                    properties=properties,
                    repetition=repetition,
                    )
                elements.append(circle)
            elif isinstance(shape, Path):
                xy = rint_cast(shape.offset + shape.vertices[0] + rep_offset)
                deltas = rint_cast(numpy.diff(shape.vertices, axis=0))
                half_width = rint_cast(shape.width / 2)
                path_type = next(k for k, v in path_cap_map.items() if v == shape.cap)    # reverse lookup
                extension_start = (path_type, shape.cap_extensions[0] if shape.cap_extensions is not None else None)
                extension_end = (path_type, shape.cap_extensions[1] if shape.cap_extensions is not None else None)
                path = fatrec.Path(
                    layer=layer,
                    datatype=datatype,
                    point_list=cast(Sequence[Sequence[int]], deltas),
                    half_width=cast(int, half_width),
                    x=xy[0],
                    y=xy[1],
                    extension_start=extension_start,       # TODO implement multiple cap types?
                    extension_end=extension_end,
                    properties=properties,
                    repetition=repetition,
                    )
                elements.append(path)
            else:
                for polygon in shape.to_polygons():
                    xy = rint_cast(polygon.offset + polygon.vertices[0] + rep_offset)
                    points = rint_cast(numpy.diff(polygon.vertices, axis=0))
                    elements.append(fatrec.Polygon(
                        layer=layer,
                        datatype=datatype,
                        x=xy[0],
                        y=xy[1],
                        point_list=cast(list[list[int]], points),
                        properties=properties,
                        repetition=repetition,
                        ))
    return elements


def _labels_to_texts(
        labels: dict[layer_t, list[Label]],
        layer2oas: Callable[[layer_t], tuple[int, int]],
        ) -> list[fatrec.Text]:
    texts = []
    for mlayer, lseq in labels.items():
        layer, datatype = layer2oas(mlayer)
        for label in lseq:
            repetition, rep_offset = repetition_masq2fata(label.repetition)
            xy = rint_cast(label.offset + rep_offset)
            properties = annotations_to_properties(label.annotations)
            texts.append(fatrec.Text(
                layer=layer,
                datatype=datatype,
                x=xy[0],
                y=xy[1],
                string=label.string,
                properties=properties,
                repetition=repetition,
                ))
    return texts


def repetition_fata2masq(
        rep: fatamorgana.GridRepetition | fatamorgana.ArbitraryRepetition | None,
        ) -> Repetition | None:
    mrep: Repetition | None
    if isinstance(rep, fatamorgana.GridRepetition):
        mrep = Grid(a_vector=rep.a_vector,
                    b_vector=rep.b_vector,
                    a_count=rep.a_count,
                    b_count=rep.b_count)
    elif isinstance(rep, fatamorgana.ArbitraryRepetition):
        displacements = numpy.cumsum(numpy.column_stack((
            rep.x_displacements,
            rep.y_displacements,
            )), axis=0)
        displacements = numpy.vstack(([0, 0], displacements))
        mrep = Arbitrary(displacements)
    elif rep is None:
        mrep = None
    return mrep


def repetition_masq2fata(
        rep: Repetition | None,
        ) -> tuple[
            fatamorgana.GridRepetition | fatamorgana.ArbitraryRepetition | None,
            tuple[int, int]
            ]:
    frep: fatamorgana.GridRepetition | fatamorgana.ArbitraryRepetition | None
    if isinstance(rep, Grid):
        a_vector = rint_cast(rep.a_vector)
        b_vector = rint_cast(rep.b_vector) if rep.b_vector is not None else None
        a_count = rint_cast(rep.a_count)
        b_count = rint_cast(rep.b_count) if rep.b_count is not None else None
        frep = fatamorgana.GridRepetition(
            a_vector=cast(list[int], a_vector),
            b_vector=cast(list[int] | None, b_vector),
            a_count=cast(int, a_count),
            b_count=cast(int | None, b_count),
            )
        offset = (0, 0)
    elif isinstance(rep, Arbitrary):
        diffs = numpy.diff(rep.displacements, axis=0)
        diff_ints = rint_cast(diffs)
        frep = fatamorgana.ArbitraryRepetition(diff_ints[:, 0], diff_ints[:, 1])        # type: ignore
        offset = rep.displacements[0, :]
    else:
        assert rep is None
        frep = None
        offset = (0, 0)
    return frep, offset


def annotations_to_properties(annotations: annotations_t) -> list[fatrec.Property]:
    #TODO determine is_standard based on key?
    properties = []
    for key, values in annotations.items():
        vals = [AString(v) if isinstance(v, str) else v
                for v in values]
        properties.append(fatrec.Property(key, vals, is_standard=False))        # type: ignore
    return properties


def properties_to_annotations(
        properties: list[fatrec.Property],
        propnames: dict[int, NString],
        propstrings: dict[int, AString],
        ) -> annotations_t:
    annotations = {}
    for proprec in properties:
        assert proprec.name is not None
        if isinstance(proprec.name, int):
            key = propnames[proprec.name].string
        else:
            key = proprec.name.string
        values: list[str | float | int] = []

        assert proprec.values is not None
        for value in proprec.values:
            if isinstance(value, (float, int)):
                values.append(value)
            elif isinstance(value, (NString, AString)):
                values.append(value.string)
            elif isinstance(value, PropStringReference):
                values.append(propstrings[value.ref].string)  # dereference
            else:
                string = repr(value)
                logger.warning(f'Converting property value for key ({key}) to string ({string})')
                values.append(string)
        annotations[key] = values
    return annotations

    properties = [fatrec.Property(key, vals, is_standard=False)
                  for key, vals in annotations.items()]
    return properties


def check_valid_names(
        names: Iterable[str],
        ) -> None:
    """
    Check all provided names to see if they're valid GDSII cell names.

    Args:
        names: Collection of names to check
        max_length: Max allowed length

    """
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + ' ')

    bad_chars = [
        name for name in names
        if not set(name).issubset(allowed_chars)
        ]

    if bad_chars:
        raise LibraryError('Names contain invalid characters:\n' + pformat(bad_chars))
