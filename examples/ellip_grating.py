# Quick script for testing arcs

import numpy

import masque
from masque import shapes


def main():
    pat = masque.Pattern()
    for rmin in numpy.arange(10, 15, 0.5):
        pat.shapes.append(shapes.Arc(
            radii=(rmin, rmin),
            width=0.1,
            angles=(-numpy.pi/4, numpy.pi/4)
        ))

    pat.visualize()


if __name__ == '__main__':
    main()
