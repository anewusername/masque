import numpy
from numpy import pi

import masque
import masque.file.gdsii
import masque.file.dxf
import masque.file.oasis
from masque import shapes, Pattern, SubPattern
from masque.repetition import Grid

from pprint import pprint


def main():
    pat = masque.Pattern(name='ellip_grating')
    for rmin in numpy.arange(10, 15, 0.5):
        pat.shapes.append(shapes.Arc(
            radii=(rmin, rmin),
            width=0.1,
            angles=(0*-numpy.pi/4, numpy.pi/4)
        ))

    pat.scale_by(1000)
#    pat.visualize()
    pat2 = pat.copy()
    pat2.name = 'grating2'

    pat3 = Pattern('sref_test')
    pat3.subpatterns = [
        SubPattern(pat, offset=(1e5, 3e5)),
        SubPattern(pat, offset=(2e5, 3e5), rotation=pi/3),
        SubPattern(pat, offset=(3e5, 3e5), rotation=pi/2),
        SubPattern(pat, offset=(4e5, 3e5), rotation=pi),
        SubPattern(pat, offset=(5e5, 3e5), rotation=3*pi/2),
        SubPattern(pat, mirrored=(True, False), offset=(1e5, 4e5)),
        SubPattern(pat, mirrored=(True, False), offset=(2e5, 4e5), rotation=pi/3),
        SubPattern(pat, mirrored=(True, False), offset=(3e5, 4e5), rotation=pi/2),
        SubPattern(pat, mirrored=(True, False), offset=(4e5, 4e5), rotation=pi),
        SubPattern(pat, mirrored=(True, False), offset=(5e5, 4e5), rotation=3*pi/2),
        SubPattern(pat, mirrored=(False, True), offset=(1e5, 5e5)),
        SubPattern(pat, mirrored=(False, True), offset=(2e5, 5e5), rotation=pi/3),
        SubPattern(pat, mirrored=(False, True), offset=(3e5, 5e5), rotation=pi/2),
        SubPattern(pat, mirrored=(False, True), offset=(4e5, 5e5), rotation=pi),
        SubPattern(pat, mirrored=(False, True), offset=(5e5, 5e5), rotation=3*pi/2),
        SubPattern(pat, mirrored=(True, True), offset=(1e5, 6e5)),
        SubPattern(pat, mirrored=(True, True), offset=(2e5, 6e5), rotation=pi/3),
        SubPattern(pat, mirrored=(True, True), offset=(3e5, 6e5), rotation=pi/2),
        SubPattern(pat, mirrored=(True, True), offset=(4e5, 6e5), rotation=pi),
        SubPattern(pat, mirrored=(True, True), offset=(5e5, 6e5), rotation=3*pi/2),
        ]

    pprint(pat3)
    pprint(pat3.subpatterns)
    pprint(pat.shapes)

    rep = Grid(a_vector=[1e4, 0],
               b_vector=[0, 1.5e4],
               a_count=3,
               b_count=2,)
    pat4 = Pattern('aref_test')
    pat4.subpatterns = [
        SubPattern(pat, repetition=rep, offset=(1e5, 3e5)),
        SubPattern(pat, repetition=rep, offset=(2e5, 3e5), rotation=pi/3),
        SubPattern(pat, repetition=rep, offset=(3e5, 3e5), rotation=pi/2),
        SubPattern(pat, repetition=rep, offset=(4e5, 3e5), rotation=pi),
        SubPattern(pat, repetition=rep, offset=(5e5, 3e5), rotation=3*pi/2),
        SubPattern(pat, repetition=rep, mirrored=(True, False), offset=(1e5, 4e5)),
        SubPattern(pat, repetition=rep, mirrored=(True, False), offset=(2e5, 4e5), rotation=pi/3),
        SubPattern(pat, repetition=rep, mirrored=(True, False), offset=(3e5, 4e5), rotation=pi/2),
        SubPattern(pat, repetition=rep, mirrored=(True, False), offset=(4e5, 4e5), rotation=pi),
        SubPattern(pat, repetition=rep, mirrored=(True, False), offset=(5e5, 4e5), rotation=3*pi/2),
        SubPattern(pat, repetition=rep, mirrored=(False, True), offset=(1e5, 5e5)),
        SubPattern(pat, repetition=rep, mirrored=(False, True), offset=(2e5, 5e5), rotation=pi/3),
        SubPattern(pat, repetition=rep, mirrored=(False, True), offset=(3e5, 5e5), rotation=pi/2),
        SubPattern(pat, repetition=rep, mirrored=(False, True), offset=(4e5, 5e5), rotation=pi),
        SubPattern(pat, repetition=rep, mirrored=(False, True), offset=(5e5, 5e5), rotation=3*pi/2),
        SubPattern(pat, repetition=rep, mirrored=(True, True), offset=(1e5, 6e5)),
        SubPattern(pat, repetition=rep, mirrored=(True, True), offset=(2e5, 6e5), rotation=pi/3),
        SubPattern(pat, repetition=rep, mirrored=(True, True), offset=(3e5, 6e5), rotation=pi/2),
        SubPattern(pat, repetition=rep, mirrored=(True, True), offset=(4e5, 6e5), rotation=pi),
        SubPattern(pat, repetition=rep, mirrored=(True, True), offset=(5e5, 6e5), rotation=3*pi/2),
        ]

    folder = 'layouts/'
    masque.file.gdsii.writefile((pat, pat2, pat3, pat4), folder + 'rep.gds.gz', 1e-9, 1e-3)

    cells = list(masque.file.gdsii.readfile(folder + 'rep.gds.gz')[0].values())
    masque.file.gdsii.writefile(cells, folder + 'rerep.gds.gz', 1e-9, 1e-3)

    masque.file.dxf.writefile(pat4, folder + 'rep.dxf.gz')
    dxf, info = masque.file.dxf.readfile(folder + 'rep.dxf.gz')
    masque.file.dxf.writefile(dxf, folder + 'rerep.dxf.gz')

    layer_map = {'base': (0,0), 'mylabel': (1,2)}
    masque.file.oasis.writefile((pat, pat2, pat3, pat4), folder + 'rep.oas.gz', 1000, layer_map=layer_map)
    oas, info = masque.file.oasis.readfile(folder + 'rep.oas.gz')
    masque.file.oasis.writefile(list(oas.values()), folder + 'rerep.oas.gz', 1000, layer_map=layer_map)
    print(info)


if __name__ == '__main__':
    main()
