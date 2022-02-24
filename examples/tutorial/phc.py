from typing import Tuple, Sequence

import numpy        # type: ignore
from numpy import pi

from masque import layer_t, Pattern, SubPattern, Label
from masque.shapes import Polygon, Circle
from masque.builder import Device, Port
from masque.library import Library, DeviceLibrary
from masque.file.gdsii import writefile

import pcgen
import basic


def perturbed_l3(
        lattice_constant: float,
        hole: Pattern,
        trench_dose: float = 1.0,
        trench_layer: layer_t = (1, 0),
        shifts_a: Sequence[float] = (0.15, 0, 0.075),
        shifts_r: Sequence[float] = (1.0, 1.0, 1.0),
        xy_size: Tuple[int, int] = (10, 10),
        perturbed_radius: float = 1.1,
        trench_width: float = 1200,
        ) -> Device:
    """
    Generate a `Device` representing a perturbed L3 cavity.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: `Pattern` object containing a single hole
        trench_dose: Dose for the trenches. Default 1.0. (Hole dose is 1.0.)
        trench_layer: Layer for the trenches, default `(1, 0)`.
        shifts_a: passed to `pcgen.l3_shift`; specifies lattice constant
            (1 - multiplicative factor) for shifting holes adjacent to
            the defect (same row). Default `(0.15, 0, 0.075)` for first,
            second, third holes.
        shifts_r: passed to `pcgen.l3_shift`; specifies radius for perturbing
            holes adjacent to the defect (same row). Default 1.0 for all holes.
            Provided sequence should have same length as `shifts_a`.
        xy_size: `(x, y)` number of mirror periods in each direction; total size is
                `2 * n + 1` holes in each direction. Default (10, 10).
        perturbed_radius: radius of holes perturbed to form an upwards-driected beam
                (multiplicative factor). Default 1.1.
        trench width: Width of the undercut trenches. Default 1200.

    Returns:
        `Device` object representing the L3 design.
    """
    xyr = pcgen.l3_shift_perturbed_defect(mirror_dims=xy_size,
                                          perturbed_radius=perturbed_radius,
                                          shifts_a=shifts_a,
                                          shifts_r=shifts_r)

    pat = Pattern(f'L3p-a{lattice_constant:g}rp{perturbed_radius:g}')
    pat.subpatterns += [SubPattern(hole,
                                   offset=(lattice_constant * x,
                                           lattice_constant * y),
                                   scale=r)
                        for x, y, r in xyr]

    min_xy, max_xy = pat.get_bounds()
    trench_dx = max_xy[0] - min_xy[0]

    pat.shapes += [
        Polygon.rect(ymin=max_xy[1], xmin=min_xy[0], lx=trench_dx, ly=trench_width,
                     layer=trench_layer, dose=trench_dose),
        Polygon.rect(ymax=min_xy[1], xmin=min_xy[0], lx=trench_dx, ly=trench_width,
                     layer=trench_layer, dose=trench_dose),
        ]

    extent = lattice_constant * xy_size[0]
    ports = {
        'input': Port((-extent, 0), rotation=0, ptype='pcwg'),
        'output': Port((extent, 0), rotation=pi, ptype='pcwg'),
        }

    return Device(pat, ports)


def waveguide(
        lattice_constant: float,
        hole: Pattern,
        length: int,
        mirror_periods: int,
        ) -> Device:
    """
    Generate a `Device` representing a photonic crystal line-defect waveguide.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: `Pattern` object containing a single hole
        length: Distance (number of mirror periods) between the input and output ports.
            Ports are placed at lattice sites.
        mirror_periods: Number of hole rows on each side of the line defect

    Returns:
        `Device` object representing the waveguide.
    """
    xy = pcgen.waveguide(length=length, num_mirror=mirror_periods)

    pat = Pattern(f'_wg-a{lattice_constant:g}l{length}')
    pat.subpatterns += [SubPattern(hole, offset=(lattice_constant * x,
                                                 lattice_constant * y))
                        for x, y in xy]

    extent = lattice_constant * length / 2
    ports = {
        'left': Port((-extent, 0), rotation=0, ptype='pcwg'),
        'right': Port((extent, 0), rotation=pi, ptype='pcwg'),
        }
    return Device(pat, ports)


def bend(
        lattice_constant: float,
        hole: Pattern,
        mirror_periods: int,
        ) -> Device:
    """
    Generate a `Device` representing a 60-degree counterclockwise bend in a photonic crystal
    line-defect waveguide.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: `Pattern` object containing a single hole
        mirror_periods: Minimum number of mirror periods on each side of the line defect.

    Returns:
        `Device` object representing the waveguide bend.
        Ports are named 'left' (input) and 'right' (output).
    """
    xy = pcgen.wgbend(num_mirror=mirror_periods)

    pat= Pattern(f'_wgbend-a{lattice_constant:g}l{mirror_periods}')
    pat.subpatterns += [SubPattern(hole, offset=(lattice_constant * x,
                                                 lattice_constant * y))
                        for x, y in xy]

    extent = lattice_constant * mirror_periods
    ports = {
        'left': Port((-extent, 0), rotation=0, ptype='pcwg'),
        'right': Port((extent / 2,
                       extent * numpy.sqrt(3) / 2), rotation=pi * 4 / 3, ptype='pcwg'),
        }
    return Device(pat, ports)


def y_splitter(
        lattice_constant: float,
        hole: Pattern,
        mirror_periods: int,
        ) -> Device:
    """
    Generate a `Device` representing a photonic crystal line-defect waveguide y-splitter.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: `Pattern` object containing a single hole
        mirror_periods: Minimum number of mirror periods on each side of the line defect.

    Returns:
        `Device` object representing the y-splitter.
        Ports are named 'in', 'top', and 'bottom'.
    """
    xy = pcgen.y_splitter(num_mirror=mirror_periods)

    pat = Pattern(f'_wgsplit_half-a{lattice_constant:g}l{mirror_periods}')
    pat.subpatterns += [SubPattern(hole, offset=(lattice_constant * x,
                                                 lattice_constant * y))
                        for x, y in xy]

    extent = lattice_constant * mirror_periods
    ports = {
        'in': Port((-extent, 0), rotation=0, ptype='pcwg'),
        'top': Port((extent / 2,
                     extent * numpy.sqrt(3) / 2), rotation=pi * 4 / 3, ptype='pcwg'),
        'bot': Port((extent / 2,
                    -extent * numpy.sqrt(3) / 2), rotation=pi * 2 / 3, ptype='pcwg'),
        }
    return Device(pat, ports)


def label_ports(device: Device, layer: layer_t = (3, 0)) -> Device:
    """
    Place a text label at each port location, specifying the port data.

    This can be used to debug port locations or to automatically generate ports
    when reading in a GDS file.

    Args:
        device: The device which is to have its ports labeled.
        layer: The layer on which the labels will be placed.

    Returns:
        `device` is returned (and altered in-place)
    """
    for name, port in device.ports.items():
        angle_deg = numpy.rad2deg(port.rotation)
        device.pattern.labels += [
            Label(string=f'{name} (angle {angle_deg:g})', layer=layer, offset=port.offset)
            ]
    return device


def main():
    a = 512
    radius = a / 2 * 0.75
    smile = basic.smile(radius)
    hole = basic.hole(radius)

    wg10 = label_ports(waveguide(lattice_constant=a, hole=hole, length=10, mirror_periods=5))
    wg05 = label_ports(waveguide(lattice_constant=a, hole=hole, length=5, mirror_periods=5))
    wg28 = label_ports(waveguide(lattice_constant=a, hole=hole, length=28, mirror_periods=5))
    bend0 = label_ports(bend(lattice_constant=a, hole=hole, mirror_periods=5))
    l3cav = label_ports(perturbed_l3(lattice_constant=a, hole=smile, xy_size=(4, 10)))
    ysplit = label_ports(y_splitter(lattice_constant=a, hole=hole, mirror_periods=5))

    dev = Device(name='my_bend', ports={})
    dev.place(wg10, offset=(0, 0), port_map={'left': 'in', 'right': 'signal'})
    dev.plug(wg10, {'signal': 'left'})
    dev.plug(ysplit, {'signal': 'in'}, {'top': 'signal1', 'bot': 'signal2'})

    dev.plug(wg05, {'signal1': 'left'})
    dev.plug(wg05, {'signal2': 'left'})
    dev.plug(bend0, {'signal1': 'right'})
    dev.plug(bend0, {'signal2': 'left'})

    dev.plug(wg10, {'signal1': 'left'})
    dev.plug(l3cav, {'signal1': 'input'})
    dev.plug(wg10, {'signal1': 'left'})

    dev.plug(wg28, {'signal2': 'left'})

    dev.plug(bend0, {'signal1': 'right'})
    dev.plug(bend0, {'signal2': 'left'})
    dev.plug(wg05, {'signal1': 'left'})
    dev.plug(wg05, {'signal2': 'left'})

    dev.plug(ysplit, {'signal1': 'bot', 'signal2': 'top'}, {'in': 'signal_out'})
    dev.plug(wg10, {'signal_out': 'left'})

    writefile(dev.pattern, 'phc.gds', 1e-9, 1e-3)
    dev.pattern.visualize()


if __name__ == '__main__':
    main()
