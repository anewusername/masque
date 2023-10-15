from typing import Sequence, Mapping

import numpy
from numpy import pi

from masque import (
    layer_t, Pattern, Ref, Label, Builder, Port, Polygon,
    Library, ILibraryView,
    )
from masque.utils import ports2data
from masque.file.gdsii import writefile, check_valid_names

import pcgen
import basic_shapes
from basic_shapes import GDS_OPTS


LATTICE_CONSTANT = 512
RADIUS = LATTICE_CONSTANT / 2 * 0.75


def ports_to_data(pat: Pattern) -> Pattern:
    """
    Bake port information into the pattern.
    This places a label at each port location on layer (3, 0) with text content
      'name:ptype angle_deg'
    """
    return ports2data.ports_to_data(pat, layer=(3, 0))


def data_to_ports(lib: Mapping[str, Pattern], name: str, pat: Pattern) -> Pattern:
    """
    Scan the Pattern to determine port locations. Same port format as `ports_to_data`
    """
    return ports2data.data_to_ports(layers=[(3, 0)], library=lib, pattern=pat, name=name)


def perturbed_l3(
        lattice_constant: float,
        hole: str,
        hole_lib: Mapping[str, Pattern],
        trench_layer: layer_t = (1, 0),
        shifts_a: Sequence[float] = (0.15, 0, 0.075),
        shifts_r: Sequence[float] = (1.0, 1.0, 1.0),
        xy_size: tuple[int, int] = (10, 10),
        perturbed_radius: float = 1.1,
        trench_width: float = 1200,
        ) -> Pattern:
    """
    Generate a `Pattern` representing a perturbed L3 cavity.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: name of a `Pattern` containing a single hole
        hole_lib: Library which contains the `Pattern` object for hole.
            Necessary because we need to know how big it is...
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
        `Pattern` object representing the L3 design.
    """
    print('Generating perturbed L3...')

    # Get hole positions and radii
    xyr = pcgen.l3_shift_perturbed_defect(mirror_dims=xy_size,
                                          perturbed_radius=perturbed_radius,
                                          shifts_a=shifts_a,
                                          shifts_r=shifts_r)

    # Build L3 cavity, using references to the provided hole pattern
    pat = Pattern()
    pat.refs[hole] += [
        Ref(scale=r, offset=(lattice_constant * x,
                             lattice_constant * y))
        for x, y, r in xyr]

    # Add rectangular undercut aids
    min_xy, max_xy = pat.get_bounds_nonempty(hole_lib)
    trench_dx = max_xy[0] - min_xy[0]

    pat.shapes[trench_layer] += [
        Polygon.rect(ymin=max_xy[1], xmin=min_xy[0], lx=trench_dx, ly=trench_width),
        Polygon.rect(ymax=min_xy[1], xmin=min_xy[0], lx=trench_dx, ly=trench_width),
        ]

    # Ports are at outer extents of the device (with y=0)
    extent = lattice_constant * xy_size[0]
    pat.ports = dict(
        input=Port((-extent, 0), rotation=0, ptype='pcwg'),
        output=Port((extent, 0), rotation=pi, ptype='pcwg'),
        )

    ports_to_data(pat)
    return pat


def waveguide(
        lattice_constant: float,
        hole: str,
        length: int,
        mirror_periods: int,
        ) -> Pattern:
    """
    Generate a `Pattern` representing a photonic crystal line-defect waveguide.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: name of a `Pattern` containing a single hole
        length: Distance (number of mirror periods) between the input and output ports.
            Ports are placed at lattice sites.
        mirror_periods: Number of hole rows on each side of the line defect

    Returns:
        `Pattern` object representing the waveguide.
    """
    # Generate hole locations
    xy = pcgen.waveguide(length=length, num_mirror=mirror_periods)

    # Build the pattern
    pat = Pattern()
    pat.refs[hole] += [
        Ref(offset=(lattice_constant * x,
                    lattice_constant * y))
        for x, y in xy]

    # Ports are at outer edges, with y=0
    extent = lattice_constant * length / 2
    pat.ports = dict(
        left=Port((-extent, 0), rotation=0, ptype='pcwg'),
        right=Port((extent, 0), rotation=pi, ptype='pcwg'),
        )

    ports_to_data(pat)
    return pat


def bend(
        lattice_constant: float,
        hole: str,
        mirror_periods: int,
        ) -> Pattern:
    """
    Generate a `Pattern` representing a 60-degree counterclockwise bend in a photonic crystal
    line-defect waveguide.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: name of a `Pattern` containing a single hole
        mirror_periods: Minimum number of mirror periods on each side of the line defect.

    Returns:
        `Pattern` object representing the waveguide bend.
        Ports are named 'left' (input) and 'right' (output).
    """
    # Generate hole locations
    xy = pcgen.wgbend(num_mirror=mirror_periods)

    # Build the pattern
    pat= Pattern()
    pat.refs[hole] += [
        Ref(offset=(lattice_constant * x,
                    lattice_constant * y))
        for x, y in xy]

    # Figure out port locations.
    extent = lattice_constant * mirror_periods
    pat.ports = dict(
        left=Port((-extent, 0), rotation=0, ptype='pcwg'),
        right=Port((extent / 2,
                    extent * numpy.sqrt(3) / 2),
                   rotation=pi * 4 / 3, ptype='pcwg'),
        )
    ports_to_data(pat)
    return pat


def y_splitter(
        lattice_constant: float,
        hole: str,
        mirror_periods: int,
        ) -> Pattern:
    """
    Generate a `Pattern` representing a photonic crystal line-defect waveguide y-splitter.

    Args:
        lattice_constant: Distance between nearest neighbor holes
        hole: name of a `Pattern` containing a single hole
        mirror_periods: Minimum number of mirror periods on each side of the line defect.

    Returns:
        `Pattern` object representing the y-splitter.
        Ports are named 'in', 'top', and 'bottom'.
    """
    # Generate hole locations
    xy = pcgen.y_splitter(num_mirror=mirror_periods)

    # Build pattern
    pat = Pattern()
    pat.refs[hole] += [
        Ref(offset=(lattice_constant * x,
                    lattice_constant * y))
        for x, y in xy]

    # Determine port locations
    extent = lattice_constant * mirror_periods
    pat.ports = {
        'in': Port((-extent, 0), rotation=0, ptype='pcwg'),
        'top': Port((extent / 2,  extent * numpy.sqrt(3) / 2), rotation=pi * 4 / 3, ptype='pcwg'),
        'bot': Port((extent / 2, -extent * numpy.sqrt(3) / 2), rotation=pi * 2 / 3, ptype='pcwg'),
        }

    ports_to_data(pat)
    return pat



def main(interactive: bool = True) -> None:
    # Generate some basic hole patterns
    shape_lib = {
        'smile': basic_shapes.smile(RADIUS),
        'hole': basic_shapes.hole(RADIUS),
        }

    # Build some devices
    a = LATTICE_CONSTANT

    devices = {}
    devices['wg05'] = waveguide(lattice_constant=a, hole='hole', length=5, mirror_periods=5)
    devices['wg10'] = waveguide(lattice_constant=a, hole='hole', length=10, mirror_periods=5)
    devices['wg28'] = waveguide(lattice_constant=a, hole='hole', length=28, mirror_periods=5)
    devices['wg90'] = waveguide(lattice_constant=a, hole='hole', length=90, mirror_periods=5)
    devices['bend0'] = bend(lattice_constant=a, hole='hole', mirror_periods=5)
    devices['ysplit'] = y_splitter(lattice_constant=a, hole='hole', mirror_periods=5)
    devices['l3cav'] = perturbed_l3(lattice_constant=a, hole='smile', hole_lib=shape_lib, xy_size=(4, 10))   # uses smile :)

    # Turn our dict of devices into a Library.
    # This provides some convenience functions in the future!
    lib = Library(devices)

    #
    # Build a circuit
    #
    # Create a `Builder`, and add the circuit to our library as "my_circuit".
    circ = Builder(library=lib, name='my_circuit')

    # Start by placing a waveguide. Call its ports "in" and "signal".
    circ.place('wg10', offset=(0, 0), port_map={'left': 'in', 'right': 'signal'})

    # Extend the signal path by attaching the "left" port of a waveguide.
    #   Since there is only one other port ("right") on the waveguide we
    # are attaching (wg10), it automatically inherits the name "signal".
    circ.plug('wg10', {'signal': 'left'})

    # We could have done the following instead:
    #   circ_pat = Pattern()
    #   lib['my_circuit'] = circ_pat
    #   circ_pat.place(lib.abstract('wg10'), ...)
    #   circ_pat.plug(lib.abstract('wg10'), ...)
    # but `Builder` lets us omit some of the repetition of `lib.abstract(...)`, and uses similar
    # syntax to `Pather` and `RenderPather`, which add wire/waveguide routing functionality.

    # Attach a y-splitter to the signal path.
    #   Since the y-splitter has 3 ports total, we can't auto-inherit the
    # port name, so we have to specify what we want to name the unattached
    # ports. We can call them "signal1" and "signal2".
    circ.plug('ysplit', {'signal': 'in'}, {'top': 'signal1', 'bot': 'signal2'})

    # Add a waveguide to both signal ports, inheriting their names.
    circ.plug('wg05', {'signal1': 'left'})
    circ.plug('wg05', {'signal2': 'left'})

    # Add a bend to both ports.
    #   Our bend's ports "left" and "right" refer to the original counterclockwise
    # orientation. We want the bends to turn in opposite directions, so we attach
    # the "right" port to "signal1" to bend clockwise, and the "left" port
    # to "signal2" to bend counterclockwise.
    #   We could also use `mirrored=(True, False)` to mirror one of the devices
    # and then use same device port on both paths.
    circ.plug('bend0', {'signal1': 'right'})
    circ.plug('bend0', {'signal2': 'left'})

    # We add some waveguides and a cavity to "signal1".
    circ.plug('wg10', {'signal1': 'left'})
    circ.plug('l3cav', {'signal1': 'input'})
    circ.plug('wg10', {'signal1': 'left'})

    # "signal2" just gets a single of equivalent length
    circ.plug('wg28', {'signal2': 'left'})

    # Now we bend both waveguides back towards each other
    circ.plug('bend0', {'signal1': 'right'})
    circ.plug('bend0', {'signal2': 'left'})
    circ.plug('wg05', {'signal1': 'left'})
    circ.plug('wg05', {'signal2': 'left'})

    # To join the waveguides, we attach a second y-junction.
    #   We plug "signal1" into the "bot" port, and "signal2" into the "top" port.
    # The remaining port gets named "signal_out".
    #   This operation would raise an exception if the ports did not line up
    # correctly (i.e. they required different rotations or translations of the
    # y-junction device).
    circ.plug('ysplit', {'signal1': 'bot', 'signal2': 'top'}, {'in': 'signal_out'})

    # Finally, add some more waveguide to "signal_out".
    circ.plug('wg10', {'signal_out': 'left'})

    # We can also add text labels for our circuit's ports.
    #   They will appear at the uppermost hierarchy level, while the individual
    # device ports will appear further down, in their respective cells.
    ports_to_data(circ.pattern)

    # Check if we forgot to include any patterns... ooops!
    if dangling := lib.dangling_refs():
        print('Warning: The following patterns are referenced, but not present in the'
              f' library! {dangling}')
        print('We\'ll solve this by merging in shape_lib, which contains those shapes...')

        lib.add(shape_lib)
        assert not lib.dangling_refs()

    # We can visualize the design. Usually it's easier to just view the GDS.
    if interactive:
        print('Visualizing... this step may be slow')
        circ.pattern.visualize(lib)

    #Write out to GDS, only keeping patterns referenced by our circuit (including itself)
    subtree = lib.subtree('my_circuit')     # don't include wg90, which we don't use
    check_valid_names(subtree.keys())
    writefile(subtree, 'circuit.gds', **GDS_OPTS)


if __name__ == '__main__':
    main()
