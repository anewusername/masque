# Quick script for testing arcs

import numpy

from masque.file import gdsii
from masque import Arc, Pattern


def main():
    pat = Pattern()
    layer = (0, 0)
    pat.shapes[layer].extend([
        Arc(
            radii=(rmin, rmin),
            width=0.1,
            angles=(-numpy.pi/4, numpy.pi/4),
            )
        for rmin in numpy.arange(10, 15, 0.5)]
        )

    pat.label(string='grating centerline', offset=(1, 0), layer=(1, 2))

    pat.scale_by(1000)
    pat.visualize()

    lib = {
        'ellip_grating': pat,
        'grating2': pat.copy(),
        }

    gdsii.writefile(lib, 'out.gds.gz', meters_per_unit=1e-9, logical_units_per_unit=1e-3)


if __name__ == '__main__':
    main()
