"""
SVG file format readers and writers
"""
from typing import Dict, Optional
import warnings

import numpy        # type: ignore
import svgwrite     # type: ignore

from .utils import mangle_name
from .. import Pattern


def writefile(pattern: Pattern,
              filename: str,
              custom_attributes: bool=False):
    """
    Write a Pattern to an SVG file, by first calling .polygonize() on it
     to change the shapes into polygons, and then writing patterns as SVG
     groups (<g>, inside <defs>), polygons as paths (<path>), and subpatterns
     as <use> elements.

    Note that this function modifies the Pattern.

    If `custom_attributes` is `True`, non-standard `pattern_layer` and `pattern_dose` attributes
     are written to the relevant elements.

    It is often a good idea to run `pattern.subpatternize()` on pattern prior to
     calling this function, especially if calling `.polygonize()` will result in very
     many vertices.

    If you want pattern polygonized with non-default arguments, just call `pattern.polygonize()`
     prior to calling this function.

    Args:
        pattern: Pattern to write to file. Modified by this function.
        filename: Filename to write to.
        custom_attributes: Whether to write non-standard `pattern_layer` and
            `pattern_dose` attributes to the SVG elements.
    """

    # Polygonize pattern
    pattern.polygonize()

    bounds = pattern.get_bounds()
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

    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {**(pattern.referenced_patterns_by_id()), id(pattern): pattern}        # type: Dict[int, Optional[Pattern]]

    # Now create a group for each row in sd_table (ie, each pattern + dose combination)
    #  and add in any Boundary and Use elements
    for pat in patterns_by_id.values():
        if pat is None:
            continue
        svg_group = svg.g(id=mangle_name(pat), fill='blue', stroke='red')

        for shape in pat.shapes:
            for polygon in shape.to_polygons():
                path_spec = poly2path(polygon.vertices + polygon.offset)

                path = svg.path(d=path_spec)
                if custom_attributes:
                    path['pattern_layer'] = polygon.layer
                    path['pattern_dose'] = polygon.dose

                svg_group.add(path)

        for subpat in pat.subpatterns:
            if subpat.pattern is None:
                continue
            transform = f'scale({subpat.scale:g}) rotate({subpat.rotation:g}) translate({subpat.offset[0]:g},{subpat.offset[1]:g})'
            use = svg.use(href='#' + mangle_name(subpat.pattern), transform=transform)
            if custom_attributes:
                use['pattern_dose'] = subpat.dose
            svg_group.add(use)

        svg.defs.add(svg_group)
    svg.add(svg.use(href='#' + mangle_name(pattern)))
    svg.save()


def writefile_inverted(pattern: Pattern, filename: str):
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
    # Polygonize and flatten pattern
    pattern.polygonize().flatten()

    bounds = pattern.get_bounds()
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
    for shape in pattern.shapes:
        for polygon in shape.to_polygons():
            path_spec += poly2path(polygon.vertices[::-1] + polygon.offset)

    svg.add(svg.path(d=path_spec, fill='blue', stroke='red'))
    svg.save()


def poly2path(vertices: numpy.ndarray) -> str:
    """
    Create an SVG path string from an Nx2 list of vertices.

    Args:
        vertices: Nx2 array of vertices.

    Returns:
        SVG path-string.
    """
    commands = 'M{:g},{:g} '.format(vertices[0][0], vertices[0][1])
    for vertex in vertices[1:]:
        commands += 'L{:g},{:g}'.format(vertex[0], vertex[1])
    commands += ' Z   '
    return commands
