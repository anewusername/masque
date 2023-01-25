from typing import Tuple, Sequence, Callable
from pprint import pformat

import numpy
from numpy import pi

from masque import Pattern, Builder, WrapLibrary, LazyLibrary, Library
from masque.file.gdsii import writefile, load_libraryfile

import pcgen
import basic_shapes
import devices
from devices import pat2dev, dev2pat
from basic_shapes import GDS_OPTS


def main() -> None:
    # Define a `LazyLibrary`, which provides lazy evaluation for generating
    #   patterns and lazy-loading of GDS contents.
    lib = LazyLibrary()

    #
    # Load some devices from a GDS file
    #

    # Scan circuit.gds and prepare to lazy-load its contents
    gds_lib, _properties = load_libraryfile('circuit.gds', postprocess=pat2dev)

    # Add it into the device library by providing a way to read port info
    #   This maintains the lazy evaluation from above, so no patterns
    # are actually read yet.
    lib.add(gds_lib)

    print('Patterns loaded from GDS into library:\n' + pformat(list(lib.keys())))

    #
    # Add some new devices to the library, this time from python code rather than GDS
    #

    lib['triangle'] = lambda: basic_shapes.triangle(devices.RADIUS)
    opts = dict(
        lattice_constant = devices.LATTICE_CONSTANT,
        hole = 'triangle',
        )

    # Triangle-based variants. These are defined here, but they won't run until they're
    #   retrieved from the library.
    lib['tri_wg10'] = lambda: devices.waveguide(length=10, mirror_periods=5, **opts)
    lib['tri_wg05'] = lambda: devices.waveguide(length=5, mirror_periods=5, **opts)
    lib['tri_wg28'] = lambda: devices.waveguide(length=28, mirror_periods=5, **opts)
    lib['tri_bend0'] = lambda: devices.bend(mirror_periods=5, **opts)
    lib['tri_ysplit'] = lambda: devices.y_splitter(mirror_periods=5, **opts)
    lib['tri_l3cav'] = lambda: devices.perturbed_l3(xy_size=(4, 10), **opts, hole_lib=lib)

    #
    # Build a mixed waveguide with an L3 cavity in the middle
    #

    # Immediately start building from an instance of the L3 cavity
    circ2 = Builder(library=lib, ports='tri_l3cav')

    print(lib['wg10'].ports)
    circ2.plug(lib.abstract('wg10'), {'input': 'right'})

    abstracts = lib.abstract_view()     # Alternate way to get abstracts
    circ2.plug(abstracts['wg10'], {'output': 'left'})
    circ2.plug(abstracts['tri_wg10'], {'input': 'right'})
    circ2.plug(abstracts['tri_wg10'], {'output': 'left'})

    # Add the circuit to the device library.
    #  It has already been generated, so we can use `set_const` as a shorthand for
    #       `lib['mixed_wg_cav'] = lambda: circ2.pattern`
    lib.set_const('mixed_wg_cav', circ2.pattern)


    #
    # Build a device that could plug into our mixed_wg_cav and joins the two ports
    #

    # We'll be designing against an existing device's interface...
    circ3 = Builder.interface(source=circ2)

    # ... that lets us continue from where we left off.
    circ3.plug(abstracts['tri_bend0'], {'input': 'right'})
    circ3.plug(abstracts['tri_bend0'], {'input': 'left'}, mirrored=(True, False)) # mirror since no tri y-symmetry
    circ3.plug(abstracts['tri_bend0'], {'input': 'right'})
    circ3.plug(abstracts['bend0'], {'output': 'left'})
    circ3.plug(abstracts['bend0'], {'output': 'left'})
    circ3.plug(abstracts['bend0'], {'output': 'left'})
    circ3.plug(abstracts['tri_wg10'], {'input': 'right'})
    circ3.plug(abstracts['tri_wg28'], {'input': 'right'})
    circ3.plug(abstracts['tri_wg10'], {'input': 'right', 'output': 'left'})

    lib.set_const('loop_segment', circ3.pattern)

    #
    # Write all devices into a GDS file
    #
    print('Writing library to file...')
    writefile(lib, 'library.gds', **GDS_OPTS)


if __name__ == '__main__':
    main()


#
#class prout:
#    def place(
#            self,
#            other: Pattern,
#            label_layer: layer_t = 'WATLAYER',
#            *,
#            port_map: Optional[Dict[str, Optional[str]]] = None,
#            **kwargs,
#            ) -> 'prout':
#
#        Pattern.place(self, other, port_map=port_map, **kwargs)
#        name: Optional[str]
#        for name in other.ports:
#            if port_map:
#                assert(name is not None)
#                name = port_map.get(name, name)
#            if name is None:
#                continue
#            self.pattern.labels += [
#                Label(string=name, offset=self.ports[name].offset, layer=layer)]
#        return self
#
