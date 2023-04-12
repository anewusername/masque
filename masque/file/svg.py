"""
SVG file format readers and writers
"""
from typing import Mapping
import warnings

import numpy
from numpy.typing import ArrayLike
import svgwrite     # type: ignore

from .utils import mangle_name
from .. import Pattern


def writefile(
        library: Mapping[str, Pattern],
        top: str,
        filename: str,
        custom_attributes: bool = False,
        ) -> None:
    """
    Write a Pattern to an SVG file, by first calling .polygonize() on it
     to change the shapes into polygons, and then writing patterns as SVG
     groups (<g>, inside <defs>), polygons as paths (<path>), and refs
     as <use> elements.

    Note that this function modifies the Pattern.

    If `custom_attributes` is `True`, a non-standard `pattern_layer` attribute
     is written to the relevant elements.

    It is often a good idea to run `pattern.dedup()` on pattern prior to
     calling this function, especially if calling `.polygonize()` will result in very
     many vertices.

    If you want pattern polygonized with non-default arguments, just call `pattern.polygonize()`
     prior to calling this function.

    Args:
        pattern: Pattern to write to file. Modified by this function.
        filename: Filename to write to.
        custom_attributes: Whether to write non-standard `pattern_layer` attribute to the
            SVG elements.
    """
    pattern = library[top]

    # Polygonize pattern
    pattern.polygonize()

    bounds = pattern.get_bounds(library=library)
    if bounds is None:
        bounds_min, bounds_max = numpy.array([[-1, -1], [1, 1]])
        warnings.warn('Pattern had no bounds (empty?); setting arbitrary viewbox')
    else:
        bounds_min, bounds_max = bounds

    viewbox = numpy.hstack((bounds_min - 1, (bounds_max - bounds_min) + 2))
    viewbox_string = '{:g} {:g} {:g} {:g}'.format(*viewbox)

    # Create file
    svg = svgwrite.Drawing(filename, profile='full', viewBox=viewbox_string,
                           debug=(not custom_attributes))

    # Now create a group for each pattern and add in any Boundary and Use elements
    for name, pat in library.items():
        svg_group = svg.g(id=mangle_name(name), fill='blue', stroke='red')

        for layer, shapes in pat.shapes.items():
            for shape in shapes:
                for polygon in shape.to_polygons():
                    path_spec = poly2path(polygon.vertices + polygon.offset)

                    path = svg.path(d=path_spec)
                    if custom_attributes:
                        path['pattern_layer'] = layer

                    svg_group.add(path)

        for target, refs in pat.refs.items():
            if target is None:
                continue
            for ref in refs:
                transform = f'scale({ref.scale:g}) rotate({ref.rotation:g}) translate({ref.offset[0]:g},{ref.offset[1]:g})'
                use = svg.use(href='#' + mangle_name(target), transform=transform)
                svg_group.add(use)

        svg.defs.add(svg_group)
    svg.add(svg.use(href='#' + mangle_name(top)))
    svg.save()


def writefile_inverted(
        library: Mapping[str, Pattern],
        top: str,
        filename: str,
        ) -> None:
    """
    Write an inverted Pattern to an SVG file, by first calling `.polygonize()` and
     `.flatten()` on it to change the shapes into polygons, then drawing a bounding
     box and drawing the polygons with reverse vertex order inside it, all within
     one `<path>` element.

    Note that this function modifies the Pattern.

    If you want pattern polygonized with non-default arguments, just call `pattern.polygonize()`
     prior to calling this function.

    Args:
        pattern: Pattern to write to file. Modified by this function.
        filename: Filename to write to.
    """
    pattern = library[top]

    # Polygonize and flatten pattern
    pattern.polygonize().flatten(library)

    bounds = pattern.get_bounds(library=library)
    if bounds is None:
        bounds_min, bounds_max = numpy.array([[-1, -1], [1, 1]])
        warnings.warn('Pattern had no bounds (empty?); setting arbitrary viewbox')
    else:
        bounds_min, bounds_max = bounds

    viewbox = numpy.hstack((bounds_min - 1, (bounds_max - bounds_min) + 2))
    viewbox_string = '{:g} {:g} {:g} {:g}'.format(*viewbox)

    # Create file
    svg = svgwrite.Drawing(filename, profile='full', viewBox=viewbox_string)

    # Draw bounding box
    slab_edge = [[bounds_min[0] - 1, bounds_max[1] + 1],
                 [bounds_max[0] + 1, bounds_max[1] + 1],
                 [bounds_max[0] + 1, bounds_min[1] - 1],
                 [bounds_min[0] - 1, bounds_min[1] - 1]]
    path_spec = poly2path(slab_edge)

    # Draw polygons with reversed vertex order
    for _layer, shapes in pattern.shapes.items():
        for shape in shapes:
            for polygon in shape.to_polygons():
                path_spec += poly2path(polygon.vertices[::-1] + polygon.offset)

    svg.add(svg.path(d=path_spec, fill='blue', stroke='red'))
    svg.save()


def poly2path(vertices: ArrayLike) -> str:
    """
    Create an SVG path string from an Nx2 list of vertices.

    Args:
        vertices: Nx2 array of vertices.

    Returns:
        SVG path-string.
    """
    verts = numpy.array(vertices, copy=False)
    commands = 'M{:g},{:g} '.format(verts[0][0], verts[0][1])
    for vertex in verts[1:]:
        commands += 'L{:g},{:g}'.format(vertex[0], vertex[1])
    commands += ' Z   '
    return commands
