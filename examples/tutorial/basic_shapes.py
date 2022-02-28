from typing import Tuple, Sequence

import numpy
from numpy import pi

from masque import layer_t, Pattern, SubPattern, Label
from masque.shapes import Circle, Arc, Polygon
from masque.builder import Device, Port
from masque.library import Library, DeviceLibrary
import masque.file.gdsii


# Note that masque units are arbitrary, and are only given
# physical significance when writing to a file.
GDS_OPTS = {
    'meters_per_unit': 1e-9,         # GDS database unit, 1 nanometer
    'logical_units_per_unit': 1e-3,  # GDS display unit, 1 micron
}


def hole(
        radius: float,
        layer: layer_t = (1, 0),
        ) -> Pattern:
    """
    Generate a pattern containing a single circular hole.

    Args:
        radius: Circle radius.
        layer: Layer to draw the circle on.

    Returns:
        Pattern, named `'hole'`
    """
    pat = Pattern('hole', shapes=[
        Circle(radius=radius, offset=(0, 0), layer=layer)
        ])
    return pat


def triangle(
        radius: float,
        layer: layer_t = (1, 0),
        ) -> Pattern:
    """
    Generate a pattern containing a single triangular hole.

    Args:
        radius: Radius of circumscribed circle.
        layer: Layer to draw the circle on.

    Returns:
        Pattern, named `'triangle'`
    """
    vertices = numpy.array([
        (numpy.cos(     pi / 2), numpy.sin(     pi / 2)),
        (numpy.cos(pi + pi / 6), numpy.sin(pi + pi / 6)),
        (numpy.cos(   - pi / 6), numpy.sin(   - pi / 6)),
    ]) * radius

    pat = Pattern('triangle', shapes=[
        Polygon(offset=(0, 0), layer=layer, vertices=vertices),
        ])
    return pat


def smile(
        radius: float,
        layer: layer_t = (1, 0),
        secondary_layer: layer_t = (1, 2)
        ) -> Pattern:
    """
    Generate a pattern containing a single smiley face.

    Args:
        radius: Boundary circle radius.
        layer: Layer to draw the outer circle on.
        secondary_layer: Layer to draw eyes and smile on.

    Returns:
        Pattern, named `'smile'`
    """
    # Make an empty pattern
    pat = Pattern('smile')

    # Add all the shapes we want
    pat.shapes += [
        Circle(radius=radius, offset=(0, 0), layer=layer),   # Outer circle
        Circle(radius=radius / 10, offset=(radius / 3, radius / 3), layer=secondary_layer),
        Circle(radius=radius / 10, offset=(-radius / 3, radius / 3), layer=secondary_layer),
        Arc(radii=(radius * 2 / 3, radius * 2 / 3), # Underlying ellipse radii
            angles=(7 / 6 * pi, 11 / 6 * pi),        # Angles limiting the arc
            width=radius / 10,
            offset=(0, 0),
            layer=secondary_layer),
        ]

    return pat


def main() -> None:
    hole_pat = hole(1000)
    smile_pat = smile(1000)
    tri_pat = triangle(1000)

    units_per_meter = 1e-9
    units_per_display_unit = 1e-3

    masque.file.gdsii.writefile([hole_pat, tri_pat, smile_pat], 'basic_shapes.gds', **GDS_OPTS)

    smile_pat.visualize()


if __name__ == '__main__':
    main()
