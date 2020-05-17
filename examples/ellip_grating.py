# Quick script for testing arcs

import numpy

import masque
import masque.file.gdsii
from masque import shapes


def main():
    pat = masque.Pattern(name='ellip_grating')
    for rmin in numpy.arange(10, 15, 0.5):
        pat.shapes.append(shapes.Arc(
            radii=(rmin, rmin),
            width=0.1,
            angles=(-numpy.pi/4, numpy.pi/4),
            layer=(0, 0),
        ))

    pat.labels.append(masque.Label(string='grating centerline', offset=(1, 0), layer=(1, 2)))

    pat.scale_by(1000)
    pat.visualize()
    pat2 = pat.copy()
    pat2.name = 'grating2'

    masque.file.gdsii.writefile((pat, pat2), 'out.gds.gz', 1e-9, 1e-3)


if __name__ == '__main__':
    main()
