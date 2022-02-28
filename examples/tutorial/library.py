from typing import Tuple, Sequence, Callable
from pprint import pformat

import numpy
from numpy import pi

from masque.builder import Device
from masque.library import Library, LibDeviceLibrary
from masque.file.gdsii import writefile, load_libraryfile

import pcgen
import basic_shapes
import devices
from basic_shapes import GDS_OPTS


def main() -> None:
    # Define a `Library`-backed `DeviceLibrary`, which provides lazy evaluation
    #   for device generation code and lazy-loading of GDS contents.
    device_lib = LibDeviceLibrary()

    #
    # Load some devices from a GDS file
    #

    # Scan circuit.gds and prepare to lazy-load its contents
    pattern_lib, _properties = load_libraryfile('circuit.gds', tag='mycirc01')

    # Add it into the device library by providing a way to read port info
    #   This maintains the lazy evaluation from above, so no patterns
    # are actually read yet.
    device_lib.add_library(pattern_lib, pat2dev=devices.pat2dev)

    print('Devices loaded from GDS into library:\n' + pformat(list(device_lib.keys())))


    #
    # Add some new devices to the library, this time from python code rather than GDS
    #

    a = devices.LATTICE_CONSTANT
    tri = basic_shapes.triangle(devices.RADIUS)

    # Convenience function for adding devices
    #  This is roughly equivalent to
    #       `device_lib[name] = lambda: devices.dev2pat(fn())`
    #  but it also guarantees that the resulting pattern is named `name`.
    def add(name: str, fn: Callable[[], Device]) -> None:
        device_lib.add_device(name=name, fn=fn, dev2pat=devices.dev2pat)

    # Triangle-based variants. These are defined here, but they won't run until they're
    #   retrieved from the library.
    add('tri_wg10', lambda: devices.waveguide(lattice_constant=a, hole=tri, length=10, mirror_periods=5))
    add('tri_wg05', lambda: devices.waveguide(lattice_constant=a, hole=tri, length=5, mirror_periods=5))
    add('tri_wg28', lambda: devices.waveguide(lattice_constant=a, hole=tri, length=28, mirror_periods=5))
    add('tri_bend0', lambda: devices.bend(lattice_constant=a, hole=tri, mirror_periods=5))
    add('tri_ysplit', lambda: devices.y_splitter(lattice_constant=a, hole=tri, mirror_periods=5))
    add('tri_l3cav', lambda: devices.perturbed_l3(lattice_constant=a, hole=tri, xy_size=(4, 10)))


    #
    # Build a mixed waveguide with an L3 cavity in the middle
    #

    # Immediately start building from an instance of the L3 cavity
    circ2 = device_lib['tri_l3cav'].build('mixed_wg_cav')

    print(device_lib['wg10'].ports)
    circ2.plug(device_lib['wg10'], {'input': 'right'})
    circ2.plug(device_lib['wg10'], {'output': 'left'})
    circ2.plug(device_lib['tri_wg10'], {'input': 'right'})
    circ2.plug(device_lib['tri_wg10'], {'output': 'left'})

    # Add the circuit to the device library.
    #  It has already been generated, so we can use `set_const` as a shorthand for
    #       `device_lib['mixed_wg_cav'] = lambda: circ2`
    device_lib.set_const(circ2)


    #
    # Build a device that could plug into our mixed_wg_cav and joins the two ports
    #

    # We'll be designing against an existing device's interface...
    circ3 = circ2.as_interface('loop_segment')
    # ... that lets us continue from where we left off.
    circ3.plug(device_lib['tri_bend0'], {'input': 'right'})
    circ3.plug(device_lib['tri_bend0'], {'input': 'left'}, mirrored=(True, False)) # mirror since no tri y-symmetry
    circ3.plug(device_lib['tri_bend0'], {'input': 'right'})
    circ3.plug(device_lib['bend0'], {'output': 'left'})
    circ3.plug(device_lib['bend0'], {'output': 'left'})
    circ3.plug(device_lib['bend0'], {'output': 'left'})
    circ3.plug(device_lib['tri_wg10'], {'input': 'right'})
    circ3.plug(device_lib['tri_wg28'], {'input': 'right'})
    circ3.plug(device_lib['tri_wg10'], {'input': 'right', 'output': 'left'})

    device_lib.set_const(circ3)

    #
    # Write all devices into a GDS file
    #

    # This line could be slow, since it generates or loads many of the devices
    #  since they were not all accessed above.
    all_device_pats = [dev.pattern for dev in device_lib.values()]

    writefile(all_device_pats, 'library.gds', **GDS_OPTS)


if __name__ == '__main__':
    main()


#
#class prout:
#    def place(
#            self,
#            other: Device,
#            label_layer: layer_t = 'WATLAYER',
#            *,
#            port_map: Optional[Dict[str, Optional[str]]] = None,
#            **kwargs,
#            ) -> 'prout':
#
#        Device.place(self, other, port_map=port_map, **kwargs)
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
