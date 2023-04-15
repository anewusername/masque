"""
DXF file format readers and writers

Notes:
 * Gzip modification time is set to 0 (start of current epoch, usually 1970-01-01)
 * ezdxf sets creation time, write time, $VERSIONGUID, and $FINGERPRINTGUID
    to unique values, so byte-for-byte reproducibility is not achievable for now
"""
from typing import Any, Callable, Mapping, cast, TextIO, IO
import io
import logging
import pathlib
import gzip

import numpy
import ezdxf
from ezdxf.enums import TextEntityAlignment

from .utils import is_gzipped, tmpfile
from .. import Pattern, Ref, PatternError, Label
from ..library import ILibraryView, LibraryView, Library
from ..shapes import Shape, Polygon, Path
from ..repetition import Grid
from ..utils import rotation_matrix_2d, layer_t, normalize_mirror


logger = logging.getLogger(__name__)


logger.warning('DXF support is experimental!')


DEFAULT_LAYER = 'DEFAULT'


def write(
        library: Mapping[str, Pattern],    # TODO could allow library=None for flat DXF
        top_name: str,
        stream: TextIO,
        *,
        dxf_version='AC1024',
        ) -> None:
    """
    Write a `Pattern` to a DXF file, by first calling `.polygonize()` to change the shapes
     into polygons, and then writing patterns as DXF `Block`s, polygons as `LWPolyline`s,
     and refs as `Insert`s.

    The top level pattern's name is not written to the DXF file. Nested patterns keep their
     names.

    Layer numbers are translated as follows:
        int: 1 -> '1'
        tuple: (1, 2) -> '1.2'
        str: '1.2' -> '1.2' (no change)

    DXF does not support shape repetition (only block repeptition). Please call
    library.wrap_repeated_shapes() before writing to file.

    Other functions you may want to call:
        - `masque.file.oasis.check_valid_names(library.keys())` to check for invalid names
        - `library.dangling_refs()` to check for references to missing patterns
        - `pattern.polygonize()` for any patterns with shapes other
            than `masque.shapes.Polygon` or `masque.shapes.Path`

    Only `Grid` repetition objects with manhattan basis vectors are preserved as arrays. Since DXF
     rotations apply to basis vectors while `masque`'s rotations do not, the basis vectors of an
     array with rotated instances must be manhattan _after_ having a compensating rotation applied.

    Args:
        library: A {name: Pattern} mapping of patterns. Only `top_name` and patterns referenced
            by it are written.
        top_name: Name of the top-level pattern to write.
        stream: Stream object to write to.
    """
    #TODO consider supporting DXF arcs?
    if not isinstance(library, ILibraryView):
        if isinstance(library, dict):
            library = LibraryView(library)
        else:
            library = LibraryView(dict(library))

    pattern = library[top_name]
    subtree = library.subtree(top_name)

    # Create library
    lib = ezdxf.new(dxf_version, setup=True)
    msp = lib.modelspace()
    _shapes_to_elements(msp, pattern.shapes)
    _labels_to_texts(msp, pattern.labels)
    _mrefs_to_drefs(msp, pattern.refs)

    # Now create a block for each referenced pattern, and add in any shapes
    for name, pat in subtree.items():
        assert pat is not None
        if name == top_name:
            continue

        block = lib.blocks.new(name=name)

        _shapes_to_elements(block, pat.shapes)
        _labels_to_texts(block, pat.labels)
        _mrefs_to_drefs(block, pat.refs)

    lib.write(stream)


def writefile(
        library: Mapping[str, Pattern],
        top_name: str,
        filename: str | pathlib.Path,
        *args,
        **kwargs,
        ) -> None:
    """
    Wrapper for `dxf.write()` that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.

    Args:
        library: A {name: Pattern} mapping of patterns. Only `top_name` and patterns referenced
            by it are written.
        top_name: Name of the top-level pattern to write.
        filename: Filename to save to.
        *args: passed to `dxf.write`
        **kwargs: passed to `dxf.write`
    """
    path = pathlib.Path(filename)

    gz_stream: IO[bytes]
    with tmpfile(path) as base_stream:
        streams: tuple[Any, ...] = (base_stream,)
        if path.suffix == '.gz':
            gz_stream = cast(IO[bytes], gzip.GzipFile(filename='', mtime=0, fileobj=base_stream, mode='wb'))
            streams = (gz_stream,) + streams
        else:
            gz_stream = base_stream
        stream = io.TextIOWrapper(gz_stream)        # type: ignore
        streams = (stream,) + streams

        try:
            write(library, top_name, stream, *args, **kwargs)
        finally:
            for ss in streams:
                ss.close()


def readfile(
        filename: str | pathlib.Path,
        *args,
        **kwargs,
        ) -> tuple[Library, dict[str, Any]]:
    """
    Wrapper for `dxf.read()` that takes a filename or path instead of a stream.

    Will automatically decompress gzipped files.

    Args:
        filename: Filename to save to.
        *args: passed to `dxf.read`
        **kwargs: passed to `dxf.read`
    """
    path = pathlib.Path(filename)
    if is_gzipped(path):
        open_func: Callable = gzip.open
    else:
        open_func = open

    with open_func(path, mode='rt') as stream:
        results = read(stream, *args, **kwargs)
    return results


def read(
        stream: TextIO,
        ) -> tuple[Library, dict[str, Any]]:
    """
    Read a dxf file and translate it into a dict of `Pattern` objects. DXF `Block`s are
     translated into `Pattern` objects; `LWPolyline`s are translated into polygons, and `Insert`s
     are translated into `Ref` objects.

    If an object has no layer it is set to this module's `DEFAULT_LAYER` ("DEFAULT").

    Args:
        stream: Stream to read from.

    Returns:
        - Top level pattern
    """
    lib = ezdxf.read(stream)
    msp = lib.modelspace()

    top_name, top_pat = _read_block(msp)
    mlib = Library({top_name: top_pat})
    for bb in lib.blocks:
        if bb.name == '*Model_Space':
            continue
        name, pat = _read_block(bb)
        mlib[name] = pat

    library_info = dict(
        layers=[ll.dxfattribs() for ll in lib.layers],
        )

    return mlib, library_info


def _read_block(block) -> tuple[str, Pattern]:
    name = block.name
    pat = Pattern()
    for element in block:
        eltype = element.dxftype()
        if eltype in ('POLYLINE', 'LWPOLYLINE'):
            if eltype == 'LWPOLYLINE':
                points = numpy.array(tuple(element.lwpoints))
            else:
                points = numpy.array(tuple(element.points()))
            attr = element.dxfattribs()
            layer = attr.get('layer', DEFAULT_LAYER)

            if points.shape[1] == 2:
                raise PatternError('Invalid or unimplemented polygon?')
                #shape = Polygon()
            elif points.shape[1] > 2:
                if (points[0, 2] != points[:, 2]).any():
                    raise PatternError('PolyLine has non-constant width (not yet representable in masque!)')
                elif points.shape[1] == 4 and (points[:, 3] != 0).any():
                    raise PatternError('LWPolyLine has bulge (not yet representable in masque!)')

                width = points[0, 2]
                if width == 0:
                    width = attr.get('const_width', 0)

                shape: Path | Polygon
                if width == 0 and len(points) > 2 and numpy.array_equal(points[0], points[-1]):
                    shape = Polygon(vertices=points[:-1, :2])
                else:
                    shape = Path(width=width, vertices=points[:, :2])

            pat.shapes[layer].append(shape)

        elif eltype in ('TEXT',):
            args = dict(
                offset=numpy.array(element.get_pos()[1])[:2],
                layer=element.dxfattribs().get('layer', DEFAULT_LAYER),
                )
            string = element.dxfattribs().get('text', '')
#            height = element.dxfattribs().get('height', 0)
#            if height != 0:
#                logger.warning('Interpreting DXF TEXT as a label despite nonzero height. '
#                               'This could be changed in the future by setting a font path in the masque DXF code.')
            pat.label(string=string, **args)
#            else:
#                pat.shapes[args['layer']].append(Text(string=string, height=height, font_path=????))
        elif eltype in ('INSERT',):
            attr = element.dxfattribs()
            xscale = attr.get('xscale', 1)
            yscale = attr.get('yscale', 1)
            if abs(xscale) != abs(yscale):
                logger.warning('Masque does not support per-axis scaling; using x-scaling only!')
            scale = abs(xscale)
            mirrored, extra_angle = normalize_mirror((yscale < 0, xscale < 0))
            rotation = numpy.deg2rad(attr.get('rotation', 0)) + extra_angle

            offset = numpy.array(attr.get('insert', (0, 0, 0)))[:2]

            args = dict(
                target=attr.get('name', None),
                offset=offset,
                scale=scale,
                mirrored=mirrored,
                rotation=rotation,
                )

            if 'column_count' in attr:
                args['repetition'] = Grid(
                    a_vector=(attr['column_spacing'], 0),
                    b_vector=(0, attr['row_spacing']),
                    a_count=attr['column_count'],
                    b_count=attr['row_count'],
                    )
            pat.ref(**args)
        else:
            logger.warning(f'Ignoring DXF element {element.dxftype()} (not implemented).')
    return name, pat


def _mrefs_to_drefs(
        block: ezdxf.layouts.BlockLayout | ezdxf.layouts.Modelspace,
        refs: dict[str | None, list[Ref]],
        ) -> None:
    def mk_blockref(encoded_name: str, ref: Ref) -> None:
        rotation = numpy.rad2deg(ref.rotation) % 360
        attribs = dict(
            xscale=ref.scale,
            yscale=ref.scale * (-1 if ref.mirrored else 1),
            rotation=rotation,
            )

        rep = ref.repetition
        if rep is None:
            block.add_blockref(encoded_name, ref.offset, dxfattribs=attribs)
        elif isinstance(rep, Grid):
            a = rep.a_vector
            b = rep.b_vector if rep.b_vector is not None else numpy.zeros(2)
            rotated_a = rotation_matrix_2d(-ref.rotation) @ a
            rotated_b = rotation_matrix_2d(-ref.rotation) @ b
            if rotated_a[1] == 0 and rotated_b[0] == 0:
                attribs['column_count'] = rep.a_count
                attribs['row_count'] = rep.b_count
                attribs['column_spacing'] = rotated_a[0]
                attribs['row_spacing'] = rotated_b[1]
                block.add_blockref(encoded_name, ref.offset, dxfattribs=attribs)
            elif rotated_a[0] == 0 and rotated_b[1] == 0:
                attribs['column_count'] = rep.b_count
                attribs['row_count'] = rep.a_count
                attribs['column_spacing'] = rotated_b[0]
                attribs['row_spacing'] = rotated_a[1]
                block.add_blockref(encoded_name, ref.offset, dxfattribs=attribs)
            else:
                #NOTE: We could still do non-manhattan (but still orthogonal) grids by getting
                #       creative with counter-rotated nested patterns, but probably not worth it.
                # Instead, just break appart the grid into individual elements:
                for dd in rep.displacements:
                    block.add_blockref(encoded_name, ref.offset + dd, dxfattribs=attribs)
        else:
            for dd in rep.displacements:
                block.add_blockref(encoded_name, ref.offset + dd, dxfattribs=attribs)

    for target, rseq in refs.items():
        if target is None:
            continue
        for ref in rseq:
            mk_blockref(target, ref)


def _shapes_to_elements(
        block: ezdxf.layouts.BlockLayout | ezdxf.layouts.Modelspace,
        shapes: dict[layer_t, list[Shape]],
        polygonize_paths: bool = False,
        ) -> None:
    # Add `LWPolyline`s for each shape.
    #   Could set do paths with width setting, but need to consider endcaps.
    for layer, sseq in shapes.items():
        attribs = dict(layer=_mlayer2dxf(layer))
        for shape in sseq:
            if shape.repetition is not None:
                raise PatternError(
                    'Shape repetitions are not supported by DXF.'
                    ' Please call library.wrap_repeated_shapes() before writing to file.'
                    )

            for polygon in shape.to_polygons():
                xy_open = polygon.vertices + polygon.offset
                xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
                block.add_lwpolyline(xy_closed, dxfattribs=attribs)


def _labels_to_texts(
        block: ezdxf.layouts.BlockLayout | ezdxf.layouts.Modelspace,
        labels: dict[layer_t, list[Label]],
        ) -> None:
    for layer, lseq in labels.items():
        attribs = dict(layer=_mlayer2dxf(layer))
        for label in lseq:
            xy = label.offset
            block.add_text(
                label.string,
                dxfattribs=attribs
                ).set_placement(xy, align=TextEntityAlignment.BOTTOM_LEFT)


def _mlayer2dxf(layer: layer_t) -> str:
    if isinstance(layer, str):
        return layer
    if isinstance(layer, int):
        return str(layer)
    if isinstance(layer, tuple):
        return f'{layer[0]}.{layer[1]}'
    raise PatternError(f'Unknown layer type: {layer} ({type(layer)})')
