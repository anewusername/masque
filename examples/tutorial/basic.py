from typing import Tuple, Sequence

import numpy        # type: ignore
from numpy import pi

from masque import layer_t, Pattern, SubPattern, Label
from masque.shapes import Circle, Arc
from masque.builder import Device, Port
from masque.library import Library, DeviceLibrary
import masque.file.gdsii

import pcgen


def hole(
        radius: float,
        layer: layer_t = (1, 0),
        ) -> Pattern:
    """
    Generate a pattern containing a single circular hole.

    Args:
        layer: Layer to draw the circle on.
        radius: Circle radius.

    Returns:
        Pattern, named `'hole'`
    """
    pat = Pattern('hole', shapes=[
        Circle(radius=radius, offset=(0, 0), layer=layer)
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


def _main() -> None:
   hole_pat = hole(1000)
   smile_pat = smile(1000)

   masque.file.gdsii.writefile([hole_pat, smile_pat], 'basic.gds', 1e-9, 1e-3)

   smile_pat.visualize()


if __name__ == '__main__':
    _main()
