from typing import Sequence

import numpy
from numpy import pi

from masque import (
    layer_t, Pattern, Label, Port,
    Circle, Arc, Polygon,
    )
import masque.file.gdsii


# Note that masque units are arbitrary, and are only given
# physical significance when writing to a file.
GDS_OPTS = dict(
    meters_per_unit = 1e-9,         # GDS database unit, 1 nanometer
    logical_units_per_unit = 1e-3,  # GDS display unit, 1 micron
    )


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
        Pattern containing a circle.
    """
    pat = Pattern()
    pat.shapes[layer].append(
        Circle(radius=radius, offset=(0, 0))
        )
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
        Pattern containing a triangle
    """
    vertices = numpy.array([
        (numpy.cos(     pi / 2), numpy.sin(     pi / 2)),
        (numpy.cos(pi + pi / 6), numpy.sin(pi + pi / 6)),
        (numpy.cos(   - pi / 6), numpy.sin(   - pi / 6)),
        ]) * radius

    pat = Pattern()
    pat.shapes[layer].extend([
        Polygon(offset=(0, 0), vertices=vertices),
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
        Pattern containing a smiley face
    """
    # Make an empty pattern
    pat = Pattern()

    # Add all the shapes we want
    pat.shapes[layer] += [
        Circle(radius=radius, offset=(0, 0)),   # Outer circle
        ]

    pat.shapes[secondary_layer] += [
        Circle(radius=radius / 10, offset=(radius / 3, radius / 3)),
        Circle(radius=radius / 10, offset=(-radius / 3, radius / 3)),
        Arc(
            radii=(radius * 2 / 3, radius * 2 / 3), # Underlying ellipse radii
            angles=(7 / 6 * pi, 11 / 6 * pi),        # Angles limiting the arc
            width=radius / 10,
            offset=(0, 0),
            ),
        ]

    return pat


def main() -> None:
    lib = {}

    lib['hole'] = hole(1000)
    lib['smile'] = smile(1000)
    lib['triangle'] = triangle(1000)

    masque.file.gdsii.writefile(lib, 'basic_shapes.gds', **GDS_OPTS)

    lib['triangle'].visualize()


if __name__ == '__main__':
    main()
