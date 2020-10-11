"""
DXF file format readers and writers
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

import numpy        # type: ignore
from numpy import pi
import ezdxf        # type: ignore

from .utils import mangle_name, make_dose_table
from .. import Pattern, SubPattern, PatternError, Label, Shape
from ..shapes import Polygon, Path
from ..repetition import Grid
from ..utils import rotation_matrix_2d, get_bit, set_bit, vector2, is_scalar, layer_t
from ..utils import remove_colinear_vertices, normalize_mirror


logger = logging.getLogger(__name__)


logger.warning('DXF support is experimental and only slightly tested!')


DEFAULT_LAYER = 'DEFAULT'


def write(pattern: Pattern,
          stream: io.TextIOBase,
          *,
          modify_originals: bool = False,
          dxf_version='AC1024',
          disambiguate_func: Callable[[Iterable[Pattern]], None] = None,
          ) -> None:
    """
    Write a `Pattern` to a DXF file, by first calling `.polygonize()` to change the shapes
     into polygons, and then writing patterns as DXF `Block`s, polygons as `LWPolyline`s,
     and subpatterns as `Insert`s.

    The top level pattern's name is not written to the DXF file. Nested patterns keep their
     names.

    Layer numbers are translated as follows:
        int: 1 -> '1'
        tuple: (1, 2) -> '1.2'
        str: '1.2' -> '1.2' (no change)

    It is often a good idea to run `pattern.subpatternize()` prior to calling this function,
     especially if calling `.polygonize()` will result in very many vertices.

    If you want pattern polygonized with non-default arguments, just call `pattern.polygonize()`
     prior to calling this function.

    Only `Grid` repetition objects with manhattan basis vectors are preserved as arrays. Since DXF
     rotations apply to basis vectors while `masque`'s rotations do not, the basis vectors of an
     array with rotated instances must be manhattan _after_ having a compensating rotation applied.

    Args:
        patterns: A Pattern or list of patterns to write to the stream.
        stream: Stream object to write to.
        modify_original: If `True`, the original pattern is modified as part of the writing
            process. Otherwise, a copy is made and `deepunlock()`-ed.
            Default `False`.
        disambiguate_func: Function which takes a list of patterns and alters them
            to make their names valid and unique. Default is `disambiguate_pattern_names`.
            WARNING: No additional error checking is performed on the results.
    """
    #TODO consider supporting DXF arcs?
    if disambiguate_func is None:
        disambiguate_func = disambiguate_pattern_names

    if not modify_originals:
        pattern = pattern.deepcopy().deepunlock()

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = pattern.referenced_patterns_by_id()
    disambiguate_func(patterns_by_id.values())

    # Create library
    lib = ezdxf.new(dxf_version, setup=True)
    msp = lib.modelspace()
    _shapes_to_elements(msp, pattern.shapes)
    _labels_to_texts(msp, pattern.labels)
    _subpatterns_to_refs(msp, pattern.subpatterns)

    # Now create a block for each referenced pattern, and add in any shapes
    for pat in patterns_by_id.values():
        assert(pat is not None)
        block = lib.blocks.new(name=pat.name)

        _shapes_to_elements(block, pat.shapes)
        _labels_to_texts(block, pat.labels)
        _subpatterns_to_refs(block, pat.subpatterns)

    lib.write(stream)


def writefile(pattern: Pattern,
              filename: Union[str, pathlib.Path],
              *args,
              **kwargs,
              ) -> None:
    """
    Wrapper for `dxf.write()` that takes a filename or path instead of a stream.

    Will automatically compress the file if it has a .gz suffix.

    Args:
        pattern: `Pattern` to save
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
        results = write(pattern, stream, *args, **kwargs)
    return results


def readfile(filename: Union[str, pathlib.Path],
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


def read(stream: io.TextIOBase,
         clean_vertices: bool = True,
         ) -> Tuple[Pattern, Dict[str, Any]]:
    """
    Read a dxf file and translate it into a dict of `Pattern` objects. DXF `Block`s are
     translated into `Pattern` objects; `LWPolyline`s are translated into polygons, and `Insert`s
     are translated into `SubPattern` objects.

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

    pat = _read_block(msp, clean_vertices)
    patterns = [pat] + [_read_block(bb, clean_vertices) for bb in lib.blocks if bb.name != '*Model_Space']

    # Create a dict of {pattern.name: pattern, ...}, then fix up all subpattern.pattern entries
    #  according to the subpattern.identifier (which is deleted after use).
    patterns_dict = dict(((p.name, p) for p in patterns))
    for p in patterns_dict.values():
        for sp in p.subpatterns:
            sp.pattern = patterns_dict[sp.identifier[0]]
            del sp.identifier

    library_info = {
        'layers': [ll.dxfattribs() for ll in lib.layers]
        }

    return pat, library_info


def _read_block(block, clean_vertices: bool) -> Pattern:
    pat = Pattern(block.name)
    for element in block:
        eltype = element.dxftype()
        if eltype in ('POLYLINE', 'LWPOLYLINE'):
            if eltype == 'LWPOLYLINE':
                points = numpy.array(tuple(element.lwpoints()))
            else:
                points = numpy.array(tuple(element.points()))
            attr = element.dxfattribs()
            args = {'layer': attr.get('layer', DEFAULT_LAYER),
                   }

            if points.shape[1] == 2:
                shape = Polygon(**args)
            elif points.shape[1] > 2:
                if (points[0, 2] != points[:, 2]).any():
                    raise PatternError('PolyLine has non-constant width (not yet representable in masque!)')
                elif points.shape[1] == 4 and (points[:, 3] != 0).any():
                    raise PatternError('LWPolyLine has bulge (not yet representable in masque!)')
                else:
                    width = points[0, 2]
                    if width == 0:
                        width = attr.get('const_width', 0)

                    if width == 0 and numpy.array_equal(points[0], points[-1]):
                        shape = Polygon(**args, vertices=points[:-1, :2])
                    else:
                        shape = Path(**args, width=width, vertices=points[:, :2])

            if clean_vertices:
                try:
                    shape.clean_vertices()
                except PatternError:
                    continue

            pat.shapes.append(shape)

        elif eltype in ('TEXT',):
            args = {'offset': element.get_pos()[1][:2],
                    'layer': element.dxfattribs().get('layer', DEFAULT_LAYER),
                   }
            string = element.dxfattribs().get('text', '')
            height = element.dxfattribs().get('height', 0)
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
            rotation = attr.get('rotation', 0) * pi/180

            offset = attr.get('insert', (0, 0, 0))[:2]

            args = {
                'offset': offset,
                'scale': scale,
                'mirrored': mirrored,
                'rotation': rotation,
                'pattern': None,
                'identifier': (attr.get('name', None),),
                }

            if 'column_count' in attr:
                args['repetition'] = Grid(
                           a_vector=(attr['column_spacing'], 0),
                           b_vector=(0, attr['row_spacing']),
                           a_count=attr['column_count'],
                           b_count=attr['row_count'])
            pat.subpatterns.append(SubPattern(**args))
        else:
            logger.warning(f'Ignoring DXF element {element.dxftype()} (not implemented).')
    return pat


def _subpatterns_to_refs(block: Union[ezdxf.layouts.BlockLayout, ezdxf.layouts.Modelspace],
                         subpatterns: List[SubPattern]) -> None:
    for subpat in subpatterns:
        if subpat.pattern is None:
            continue
        encoded_name = subpat.pattern.name

        rotation = (subpat.rotation * 180 / numpy.pi) % 360
        attribs = {
            'xscale': subpat.scale * (-1 if subpat.mirrored[1] else 1),
            'yscale': subpat.scale * (-1 if subpat.mirrored[0] else 1),
            'rotation': rotation,
            }

        rep = subpat.repetition
        if rep is None:
            block.add_blockref(encoded_name, subpat.offset, dxfattribs=attribs)
        elif isinstance(rep, Grid):
            a = rep.a_vector
            b = rep.b_vector if rep.b_vector is not None else numpy.zeros(2)
            rotated_a = rotation_matrix_2d(-subpat.rotation) @ a
            rotated_b = rotation_matrix_2d(-subpat.rotation) @ b
            if rotated_a[1] == 0 and rotated_b[0] == 0:
                attribs['column_count'] = rep.a_count
                attribs['row_count'] = rep.b_count
                attribs['column_spacing'] = rotated_a[0]
                attribs['row_spacing'] = rotated_b[1]
                block.add_blockref(encoded_name, subpat.offset, dxfattribs=attribs)
            elif rotated_a[0] == 0 and rotated_b[1] == 0:
                attribs['column_count'] = rep.b_count
                attribs['row_count'] = rep.a_count
                attribs['column_spacing'] = rotated_b[0]
                attribs['row_spacing'] = rotated_a[1]
                block.add_blockref(encoded_name, subpat.offset, dxfattribs=attribs)
            else:
                #NOTE: We could still do non-manhattan (but still orthogonal) grids by getting
                #       creative with counter-rotated nested patterns, but probably not worth it.
                # Instead, just break appart the grid into individual elements:
                for dd in rep.displacements:
                    block.add_blockref(encoded_name, subpat.offset + dd, dxfattribs=attribs)
        else:
            for dd in rep.displacements:
                block.add_blockref(encoded_name, subpat.offset + dd, dxfattribs=attribs)


def _shapes_to_elements(block: Union[ezdxf.layouts.BlockLayout, ezdxf.layouts.Modelspace],
                        shapes: List[Shape],
                        polygonize_paths: bool = False):
    # Add `LWPolyline`s for each shape.
    #   Could set do paths with width setting, but need to consider endcaps.
    for shape in shapes:
        attribs = {'layer': _mlayer2dxf(shape.layer)}
        for polygon in shape.to_polygons():
            xy_open = polygon.vertices + polygon.offset
            xy_closed = numpy.vstack((xy_open, xy_open[0, :]))
            block.add_lwpolyline(xy_closed, dxfattribs=attribs)


def _labels_to_texts(block: Union[ezdxf.layouts.BlockLayout, ezdxf.layouts.Modelspace],
                     labels: List[Label]) -> None:
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


def disambiguate_pattern_names(patterns: Sequence[Pattern],
                               max_name_length: int = 32,
                               suffix_length: int = 6,
                               dup_warn_filter: Callable[[str,], bool] = None,      # If returns False, don't warn about this name
                               ) -> None:
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
            logger.warning(f'Empty pattern name saved as "{suffixed_name}"')
        elif suffixed_name != sanitized_name:
            if dup_warn_filter is None or dup_warn_filter(pat.name):
                logger.warning(f'Pattern name "{pat.name}" ({sanitized_name}) appears multiple times;\n' +
                               f' renaming to "{suffixed_name}"')

        if len(suffixed_name) == 0:
            # Should never happen since zero-length names are replaced
            raise PatternError(f'Zero-length name after sanitize,\n originally "{pat.name}"')
        if len(suffixed_name) > max_name_length:
            raise PatternError(f'Pattern name "{suffixed_name!r}" length > {max_name_length} after encode,\n' +
                               f' originally "{pat.name}"')

        pat.name = suffixed_name
        used_names.append(suffixed_name)

