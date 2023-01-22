"""
DXF file format readers and writers
"""
from typing import List, Any, Dict, Tuple, Callable, Union, Sequence, Iterable, Mapping
import re
import io
import base64
import struct
import logging
import pathlib
import gzip

import numpy
import ezdxf        # type: ignore

from .. import Pattern, Ref, PatternError, Label, Shape
from ..shapes import Polygon, Path
from ..repetition import Grid
from ..utils import rotation_matrix_2d, layer_t


logger = logging.getLogger(__name__)


logger.warning('DXF support is experimental and only slightly tested!')


DEFAULT_LAYER = 'DEFAULT'


def write(
        top_name: str,
        library: Mapping[str, Pattern],
        stream: io.TextIOBase,
        *,
        modify_originals: bool = False,
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

    It is often a good idea to run `pattern.dedup()` prior to calling this function,
     especially if calling `.polygonize()` will result in very many vertices.

    If you want pattern polygonized with non-default arguments, just call `pattern.polygonize()`
     prior to calling this function.

    Only `Grid` repetition objects with manhattan basis vectors are preserved as arrays. Since DXF
     rotations apply to basis vectors while `masque`'s rotations do not, the basis vectors of an
     array with rotated instances must be manhattan _after_ having a compensating rotation applied.

    Args:
        top_name: Name of the top-level pattern to write.
        library: A {name: Pattern} mapping of patterns. Only `top_name` and patterns referenced
            by it are written.
        stream: Stream object to write to.
        modify_original: If `True`, the original pattern is modified as part of the writing
            process. Otherwise, a copy is made.
            Default `False`.
        disambiguate_func: Function which takes a list of patterns and alters them
            to make their names valid and unique. Default is `disambiguate_pattern_names`.
            WARNING: No additional error checking is performed on the results.
    """
    #TODO consider supporting DXF arcs?

    #TODO name checking
    bad_keys = check_valid_names(library.keys())

    if not modify_originals:
        library = library.deepcopy()

    pattern = library[top_name]

    # Create library
    lib = ezdxf.new(dxf_version, setup=True)
    msp = lib.modelspace()
    _shapes_to_elements(msp, pattern.shapes)
    _labels_to_texts(msp, pattern.labels)
    _mrefs_to_drefs(msp, pattern.refs)

    # Now create a block for each referenced pattern, and add in any shapes
    for name, pat in library.items():
        assert(pat is not None)
        block = lib.blocks.new(name=name)

        _shapes_to_elements(block, pat.shapes)
        _labels_to_texts(block, pat.labels)
        _mrefs_to_drefs(block, pat.refs)

    lib.write(stream)


def writefile(
        top_name: str,
        library: Mapping[str, Pattern],
        filename: Union[str, pathlib.Path],
        *args,
        **kwargs,
        ) -> None:
    """
    Wrapper for `dxf.write()` that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.

    Args:
        top_name: Name of the top-level pattern to write.
        library: A {name: Pattern} mapping of patterns. Only `top_name` and patterns referenced
            by it are written.
        filename: Filename to save to.
        *args: passed to `dxf.write`
        **kwargs: passed to `dxf.write`
    """
    path = pathlib.Path(filename)
    if path.suffix == '.gz':
        open_func: Callable = gzip.open
    else:
        open_func = open

    with open_func(path, mode='wt') as stream:
        write(top_name, library, stream, *args, **kwargs)


def readfile(
        filename: Union[str, pathlib.Path],
        *args,
        **kwargs,
        ) -> Tuple[Pattern, Dict[str, Any]]:
    """
    Wrapper for `dxf.read()` that takes a filename or path instead of a stream.

    Will automatically decompress files with a .gz suffix.

    Args:
        filename: Filename to save to.
        *args: passed to `dxf.read`
        **kwargs: passed to `dxf.read`
    """
    path = pathlib.Path(filename)
    if path.suffix == '.gz':
        open_func: Callable = gzip.open
    else:
        open_func = open

    with open_func(path, mode='rt') as stream:
        results = read(stream, *args, **kwargs)
    return results


def read(
        stream: io.TextIOBase,
        clean_vertices: bool = True,
        ) -> Tuple[Dict[str, Pattern], Dict[str, Any]]:
    """
    Read a dxf file and translate it into a dict of `Pattern` objects. DXF `Block`s are
     translated into `Pattern` objects; `LWPolyline`s are translated into polygons, and `Insert`s
     are translated into `Ref` objects.

    If an object has no layer it is set to this module's `DEFAULT_LAYER` ("DEFAULT").

    Args:
        stream: Stream to read from.
        clean_vertices: If `True`, remove any redundant vertices when loading polygons.
            The cleaning process removes any polygons with zero area or <3 vertices.
            Default `True`.

    Returns:
        - Top level pattern
    """
    lib = ezdxf.read(stream)
    msp = lib.modelspace()

    npat = _read_block(msp, clean_vertices)
    patterns_dict = dict([npat]
        + [_read_block(bb, clean_vertices) for bb in lib.blocks if bb.name != '*Model_Space'])

    library_info = {
        'layers': [ll.dxfattribs() for ll in lib.layers]
        }

    return patterns_dict, library_info


def _read_block(block, clean_vertices: bool) -> Tuple[str, Pattern]:
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
                #shape = Polygon(layer=layer)
            elif points.shape[1] > 2:
                if (points[0, 2] != points[:, 2]).any():
                    raise PatternError('PolyLine has non-constant width (not yet representable in masque!)')
                elif points.shape[1] == 4 and (points[:, 3] != 0).any():
                    raise PatternError('LWPolyLine has bulge (not yet representable in masque!)')

                width = points[0, 2]
                if width == 0:
                    width = attr.get('const_width', 0)

                shape: Union[Path, Polygon]
                if width == 0 and len(points) > 2 and numpy.array_equal(points[0], points[-1]):
                    shape = Polygon(layer=layer, vertices=points[:-1, :2])
                else:
                    shape = Path(layer=layer, width=width, vertices=points[:, :2])

            if clean_vertices:
                try:
                    shape.clean_vertices()
                except PatternError:
                    continue

            pat.shapes.append(shape)

        elif eltype in ('TEXT',):
            args = {'offset': numpy.array(element.get_pos()[1])[:2],
                    'layer': element.dxfattribs().get('layer', DEFAULT_LAYER),
                   }
            string = element.dxfattribs().get('text', '')
#            height = element.dxfattribs().get('height', 0)
#            if height != 0:
#                logger.warning('Interpreting DXF TEXT as a label despite nonzero height. '
#                               'This could be changed in the future by setting a font path in the masque DXF code.')
            pat.labels.append(Label(string=string, **args))
#            else:
#                pat.shapes.append(Text(string=string, height=height, font_path=????))
        elif eltype in ('INSERT',):
            attr = element.dxfattribs()
            xscale = attr.get('xscale', 1)
            yscale = attr.get('yscale', 1)
            if abs(xscale) != abs(yscale):
                logger.warning('Masque does not support per-axis scaling; using x-scaling only!')
            scale = abs(xscale)
            mirrored = (yscale < 0, xscale < 0)
            rotation = numpy.deg2rad(attr.get('rotation', 0))

            offset = numpy.array(attr.get('insert', (0, 0, 0)))[:2]

            args = {
                'target': (attr.get('name', None),),
                'offset': offset,
                'scale': scale,
                'mirrored': mirrored,
                'rotation': rotation,
                'pattern': None,
                }

            if 'column_count' in attr:
                args['repetition'] = Grid(a_vector=(attr['column_spacing'], 0),
                                          b_vector=(0, attr['row_spacing']),
                                          a_count=attr['column_count'],
                                          b_count=attr['row_count'])
            pat.ref(**args)
        else:
            logger.warning(f'Ignoring DXF element {element.dxftype()} (not implemented).')
    return name, pat


def _mrefs_to_drefs(
        block: Union[ezdxf.layouts.BlockLayout, ezdxf.layouts.Modelspace],
        refs: List[Ref],
        ) -> None:
    for ref in refs:
        if ref.target is None:
            continue
        encoded_name = ref.target

        rotation = (ref.rotation * 180 / numpy.pi) % 360
        attribs = {
            'xscale': ref.scale * (-1 if ref.mirrored[1] else 1),
            'yscale': ref.scale * (-1 if ref.mirrored[0] else 1),
            'rotation': rotation,
            }

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


def _shapes_to_elements(
        block: Union[ezdxf.layouts.BlockLayout, ezdxf.layouts.Modelspace],
        shapes: List[Shape],
        polygonize_paths: bool = False,
        ) -> None:
    # Add `LWPolyline`s for each shape.
    #   Could set do paths with width setting, but need to consider endcaps.
    for shape in shapes:
        attribs = {'layer': _mlayer2dxf(shape.layer)}
        for polygon in shape.to_polygons():
            xy_open = polygon.vertices + polygon.offset
            xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
            block.add_lwpolyline(xy_closed, dxfattribs=attribs)


def _labels_to_texts(
        block: Union[ezdxf.layouts.BlockLayout, ezdxf.layouts.Modelspace],
        labels: List[Label],
        ) -> None:
    for label in labels:
        attribs = {'layer': _mlayer2dxf(label.layer)}
        xy = label.offset
        block.add_text(label.string, dxfattribs=attribs).set_pos(xy, align='BOTTOM_LEFT')


def _mlayer2dxf(layer: layer_t) -> str:
    if isinstance(layer, str):
        return layer
    if isinstance(layer, int):
        return str(layer)
    if isinstance(layer, tuple):
        return f'{layer[0]}.{layer[1]}'
    raise PatternError(f'Unknown layer type: {layer} ({type(layer)})')


def disambiguate_pattern_names(
        names: Iterable[str],
        max_name_length: int = 32,
        suffix_length: int = 6,
        ) -> List[str]:
    """
    Args:
        names: List of pattern names to disambiguate
        max_name_length: Names longer than this will be truncated
        suffix_length: Names which get truncated are truncated by this many extra characters. This is to
            leave room for a suffix if one is necessary.
    """
    new_names = []
    for name in names:
        sanitized_name = re.compile(r'[^A-Za-z0-9_\?\$]').sub('_', name)

        i = 0
        suffixed_name = sanitized_name
        while suffixed_name in new_names or suffixed_name == '':
            suffix = base64.b64encode(struct.pack('>Q', i), b'$?').decode('ASCII')

            suffixed_name = sanitized_name + '$' + suffix[:-1].lstrip('A')
            i += 1

        if sanitized_name == '':
            logger.warning(f'Empty pattern name saved as "{suffixed_name}"')

        if len(suffixed_name) == 0:
            # Should never happen since zero-length names are replaced
            raise PatternError(f'Zero-length name after sanitize,\n originally "{name}"')
        if len(suffixed_name) > max_name_length:
            raise PatternError(f'Pattern name "{suffixed_name!r}" length > {max_name_length} after encode,\n'
                               + f' originally "{name}"')

        new_names.append(suffixed_name)
    return new_names
