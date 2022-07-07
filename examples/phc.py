from typing import Tuple, Sequence

import numpy        # type: ignore
from numpy import pi

from masque import layer_t, Pattern, SubPattern, Label
from masque.shapes import Polygon, Circle
from masque.builder import Device, Port
from masque.library import Library, DeviceLibrary
from masque.file.klamath import writefile

import pcgen


HOLE_SCALE: float = 1000
''' Radius for the 'hole' cell. Should be significantly bigger than
    1 (minimum database unit) in order to have enough precision to
    reasonably represent a polygonized circle (for GDS)
'''

def hole(layer: layer_t,
         radius: float = HOLE_SCALE * 0.35,
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
        Circle(radius=radius, offset=(0, 0), layer=layer, dose=1.0)
        ])
    return pat


def perturbed_l3(lattice_constant: float,
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
    pat.subpatterns += [SubPattern(hole, offset=(lattice_constant * x,
                                                 lattice_constant * y), scale=r * lattice_constant / HOLE_SCALE)
                        for x, y, r in xyr]

    min_xy, max_xy = pat.get_bounds()
    trench_dx = max_xy[0] - min_xy[0]

    pat.shapes += [
        Polygon.rect(ymin=max_xy[1], xmin=min_xy[0], lx=trench_dx, ly=trench_width,
                     layer=trench_layer, dose=trench_dose),
        Polygon.rect(ymax=min_xy[1], xmin=min_xy[0], lx=trench_dx, ly=trench_width,
                     layer=trench_layer, dose=trench_dose),
        ]

    ports = {
        'input': Port((-lattice_constant * xy_size[0], 0), rotation=0, ptype=1),
        'output': Port((lattice_constant * xy_size[0], 0), rotation=pi, ptype=1),
        }

    return Device(pat, ports)


def waveguide(lattice_constant: float,
              hole: Pattern,
              length: int,
              mirror_periods: int,
              ) -> Device:
    xy = pcgen.waveguide(length=length + 2, num_mirror=mirror_periods)

    pat = Pattern(f'_wg-a{lattice_constant:g}l{length}')
    pat.subpatterns += [SubPattern(hole, offset=(lattice_constant * x,
                                                 lattice_constant * y), scale=lattice_constant / HOLE_SCALE)
                        for x, y in xy]

    ports = {
        'left': Port((-lattice_constant * length / 2, 0), rotation=0, ptype=1),
        'right': Port((lattice_constant * length / 2, 0), rotation=pi, ptype=1),
        }
    return Device(pat, ports)


def bend(lattice_constant: float,
         hole: Pattern,
         mirror_periods: int,
         ) -> Device:
    xy = pcgen.wgbend(num_mirror=mirror_periods)

    pat_half = Pattern(f'_wgbend_half-a{lattice_constant:g}l{mirror_periods}')
    pat_half.subpatterns += [SubPattern(hole, offset=(lattice_constant * x,
                                                      lattice_constant * y), scale=lattice_constant / HOLE_SCALE)
                             for x, y in xy]

    pat = Pattern(f'_wgbend-a{lattice_constant:g}l{mirror_periods}')
    pat.addsp(pat_half, offset=(0, 0), rotation=0, mirrored=(False, False))
    pat.addsp(pat_half, offset=(0, 0), rotation=-2 * pi / 3, mirrored=(True, False))


    ports = {
        'left': Port((-lattice_constant * mirror_periods, 0), rotation=0, ptype=1),
        'right': Port((lattice_constant * mirror_periods / 2,
                       lattice_constant * mirror_periods * numpy.sqrt(3) / 2), rotation=pi * 4 / 3, ptype=1),
        }
    return Device(pat, ports)


def label_ports(device: Device, layer: layer_t = (3, 0)) -> Device:
    for name, port in device.ports.items():
        angle_deg = numpy.rad2deg(port.rotation)
        device.pattern.labels += [
            Label(string=f'{name} (angle {angle_deg:g})', layer=layer, offset=port.offset)
            ]
    return device


def main():
    hole_layer = (1, 2)
    a = 512
    hole_pat = hole(layer=hole_layer)
    wg0 = label_ports(waveguide(lattice_constant=a, hole=hole_pat, length=10, mirror_periods=5))
    wg1 = label_ports(waveguide(lattice_constant=a, hole=hole_pat, length=5, mirror_periods=5))
    bend0 = label_ports(bend(lattice_constant=a, hole=hole_pat, mirror_periods=5))
    l3cav = label_ports(perturbed_l3(lattice_constant=a, hole=hole_pat, xy_size=(4, 10)))

    dev = Device(name='my_bend', ports={})
    dev.place(wg0, offset=(0, 0), port_map={'left': 'in', 'right': 'signal'})
    dev.plug(wg0, {'signal': 'left'})
    dev.plug(bend0, {'signal': 'left'})
    dev.plug(wg1, {'signal': 'left'})
    dev.plug(bend0, {'signal': 'right'})
    dev.plug(wg0, {'signal': 'left'})
    dev.plug(l3cav, {'signal': 'input'})
    dev.plug(wg0, {'signal': 'left'})

    writefile(dev.pattern, 'phc.gds', 1e-9, 1e-3)
    dev.pattern.visualize()


if __name__ == '__main__':
    main()
