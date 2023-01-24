from pprint import pprint
from pathlib import Path

import numpy
from numpy import pi

import masque
from masque import Pattern, Ref, Arc, WrapLibrary
from masque.repetition import Grid
from masque.file import gdsii, dxf, oasis



def main():
    lib = WrapLibrary()

    cell_name = 'ellip_grating'
    pat = masque.Pattern()
    for rmin in numpy.arange(10, 15, 0.5):
        pat.shapes.append(Arc(
            radii=(rmin, rmin),
            width=0.1,
            angles=(0 * -pi/4, pi/4),
            annotations={'1': ['blah']},
            ))

    pat.scale_by(1000)
#    pat.visualize()
    lib[cell_name] = pat
    print(f'\nAdded {cell_name}:')
    pprint(pat.shapes)

    new_name = lib.get_name(cell_name)
    lib[new_name] = pat.copy()
    print(f'\nAdded a copy of {cell_name} as {new_name}')

    pat3 = Pattern()
    pat3.refs = [
        Ref(cell_name, offset=(1e5, 3e5), annotations={'4': ['Hello I am the base Ref']}),
        Ref(cell_name, offset=(2e5, 3e5), rotation=pi/3),
        Ref(cell_name, offset=(3e5, 3e5), rotation=pi/2),
        Ref(cell_name, offset=(4e5, 3e5), rotation=pi),
        Ref(cell_name, offset=(5e5, 3e5), rotation=3*pi/2),
        Ref(cell_name, mirrored=(True, False), offset=(1e5, 4e5)),
        Ref(cell_name, mirrored=(True, False), offset=(2e5, 4e5), rotation=pi/3),
        Ref(cell_name, mirrored=(True, False), offset=(3e5, 4e5), rotation=pi/2),
        Ref(cell_name, mirrored=(True, False), offset=(4e5, 4e5), rotation=pi),
        Ref(cell_name, mirrored=(True, False), offset=(5e5, 4e5), rotation=3*pi/2),
        Ref(cell_name, mirrored=(False, True), offset=(1e5, 5e5)),
        Ref(cell_name, mirrored=(False, True), offset=(2e5, 5e5), rotation=pi/3),
        Ref(cell_name, mirrored=(False, True), offset=(3e5, 5e5), rotation=pi/2),
        Ref(cell_name, mirrored=(False, True), offset=(4e5, 5e5), rotation=pi),
        Ref(cell_name, mirrored=(False, True), offset=(5e5, 5e5), rotation=3*pi/2),
        Ref(cell_name, mirrored=(True, True), offset=(1e5, 6e5)),
        Ref(cell_name, mirrored=(True, True), offset=(2e5, 6e5), rotation=pi/3),
        Ref(cell_name, mirrored=(True, True), offset=(3e5, 6e5), rotation=pi/2),
        Ref(cell_name, mirrored=(True, True), offset=(4e5, 6e5), rotation=pi),
        Ref(cell_name, mirrored=(True, True), offset=(5e5, 6e5), rotation=3*pi/2),
        ]

    lib['sref_test'] = pat3
    print('\nAdded sref_test:')
    pprint(pat3)
    pprint(pat3.refs)

    rep = Grid(
        a_vector=[1e4, 0],
        b_vector=[0, 1.5e4],
        a_count=3,
        b_count=2,
        )
    pat4 = Pattern()
    pat4.refs = [
        Ref(cell_name, repetition=rep, offset=(1e5, 3e5)),
        Ref(cell_name, repetition=rep, offset=(2e5, 3e5), rotation=pi/3),
        Ref(cell_name, repetition=rep, offset=(3e5, 3e5), rotation=pi/2),
        Ref(cell_name, repetition=rep, offset=(4e5, 3e5), rotation=pi),
        Ref(cell_name, repetition=rep, offset=(5e5, 3e5), rotation=3*pi/2),
        Ref(cell_name, repetition=rep, mirrored=(True, False), offset=(1e5, 4e5)),
        Ref(cell_name, repetition=rep, mirrored=(True, False), offset=(2e5, 4e5), rotation=pi/3),
        Ref(cell_name, repetition=rep, mirrored=(True, False), offset=(3e5, 4e5), rotation=pi/2),
        Ref(cell_name, repetition=rep, mirrored=(True, False), offset=(4e5, 4e5), rotation=pi),
        Ref(cell_name, repetition=rep, mirrored=(True, False), offset=(5e5, 4e5), rotation=3*pi/2),
        Ref(cell_name, repetition=rep, mirrored=(False, True), offset=(1e5, 5e5)),
        Ref(cell_name, repetition=rep, mirrored=(False, True), offset=(2e5, 5e5), rotation=pi/3),
        Ref(cell_name, repetition=rep, mirrored=(False, True), offset=(3e5, 5e5), rotation=pi/2),
        Ref(cell_name, repetition=rep, mirrored=(False, True), offset=(4e5, 5e5), rotation=pi),
        Ref(cell_name, repetition=rep, mirrored=(False, True), offset=(5e5, 5e5), rotation=3*pi/2),
        Ref(cell_name, repetition=rep, mirrored=(True, True), offset=(1e5, 6e5)),
        Ref(cell_name, repetition=rep, mirrored=(True, True), offset=(2e5, 6e5), rotation=pi/3),
        Ref(cell_name, repetition=rep, mirrored=(True, True), offset=(3e5, 6e5), rotation=pi/2),
        Ref(cell_name, repetition=rep, mirrored=(True, True), offset=(4e5, 6e5), rotation=pi),
        Ref(cell_name, repetition=rep, mirrored=(True, True), offset=(5e5, 6e5), rotation=3*pi/2),
        ]

    lib['aref_test'] = pat4
    print('\nAdded aref_test')

    folder = Path('./layouts/')
    print(f'...writing files to {folder}...')

    gds1 = folder / 'rep.gds.gz'
    gds2 = folder / 'rerep.gds.gz'
    print(f'Initial write to {gds1}')
    gdsii.writefile(lib, gds1, 1e-9, 1e-3)

    print(f'Read back and rewrite to {gds2}')
    readback_lib, _info = gdsii.readfile(gds1)
    gdsii.writefile(readback_lib, gds2, 1e-9, 1e-3)

    dxf1 = folder / 'rep.dxf.gz'
    dxf2 = folder / 'rerep.dxf.gz'
    print(f'Write aref_test to {dxf1}')
    dxf.writefile(lib, 'aref_test', dxf1)

    print(f'Read back and rewrite to {dxf2}')
    dxf_lib, _info = dxf.readfile(dxf1)
    print(WrapLibrary(dxf_lib))
    dxf.writefile(dxf_lib, 'Model', dxf2)

    layer_map = {'base': (0,0), 'mylabel': (1,2)}
    oas1 = folder / 'rep.oas'
    oas2 = folder / 'rerep.oas'
    print(f'Write lib to {oas1}')
    oasis.writefile(lib, oas1, 1000, layer_map=layer_map)

    print(f'Read back and rewrite to {oas2}')
    oas_lib, oas_info = oasis.readfile(oas1)
    oasis.writefile(oas_lib, oas2, 1000, layer_map=layer_map)

    print('OASIS info:')
    pprint(oas_info)


if __name__ == '__main__':
    main()
