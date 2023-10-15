"""
Manual wire routing tutorial: Pather and BasicTool
"""
from typing import Callable
from numpy import pi
from masque import Pather, RenderPather, Library, Pattern, Port, layer_t, map_layers
from masque.builder.tools import BasicTool, PathTool
from masque.file.gdsii import writefile

from basic_shapes import GDS_OPTS

#
# Define some basic wire widths, in nanometers
# M2 is the top metal; M1 is below it and connected with vias on V1
#
M1_WIDTH = 1000
V1_WIDTH = 500
M2_WIDTH = 4000

#
# First, we can define some functions for generating our wire geometry
#

def make_pad() -> Pattern:
    """
    Create a pattern with a single rectangle of M2, with a single port on the bottom

    Every pad will be an instance of the same pattern, so we will only call this function once.
    """
    pat = Pattern()
    pat.rect(layer='M2', xctr=0, yctr=0, lx=3 * M2_WIDTH, ly=4 * M2_WIDTH)
    pat.ports['wire_port'] = Port((0, -2 * M2_WIDTH), rotation=pi / 2, ptype='m2wire')
    return pat


def make_via(
        layer_top: layer_t,
        layer_via: layer_t,
        layer_bot: layer_t,
        width_top: float,
        width_via: float,
        width_bot: float,
        ptype_top: str,
        ptype_bot: str,
        ) -> Pattern:
    """
      Generate three concentric squares, on the provided layers
    (`layer_top`, `layer_via`, `layer_bot`) and with the provided widths
    (`width_top`, `width_via`, `width_bot`).

      Two ports are added, with the provided ptypes (`ptype_top`, `ptype_bot`).
    They are placed at the left edge of the top layer and right edge of the
    bottom layer, respectively.

    We only have one via type, so we will only call this function once.
    """
    pat = Pattern()
    pat.rect(layer=layer_via, xctr=0, yctr=0, lx=width_via, ly=width_via)
    pat.rect(layer=layer_bot, xctr=0, yctr=0, lx=width_bot, ly=width_bot)
    pat.rect(layer=layer_top, xctr=0, yctr=0, lx=width_top, ly=width_top)
    pat.ports = {
        'top': Port(offset=(-width_top / 2, 0), rotation=0, ptype=ptype_top),
        'bottom': Port(offset=(width_bot / 2, 0), rotation=pi, ptype=ptype_bot),
        }
    return pat


def make_bend(layer: layer_t, width: float, ptype: str) -> Pattern:
    """
      Generate a triangular wire, with ports at the left (input) and bottom (output) edges.
    This is effectively a clockwise wire bend.

      Every bend will be the same, so we only need to call this twice (once each for M1 and M2).
    We could call it additional times for different wire widths or bend types (e.g. squares).
    """
    pat = Pattern()
    pat.polygon(layer=layer, vertices=[(0, -width / 2), (0, width / 2), (width, -width / 2)])
    pat.ports = {
        'input': Port(offset=(0, 0), rotation=0, ptype=ptype),
        'output': Port(offset=(width / 2, -width / 2), rotation=pi / 2, ptype=ptype),
        }
    return pat


def make_straight_wire(layer: layer_t, width: float, ptype: str, length: float) -> Pattern:
    """
    Generate a straight wire with ports along either end (x=0 and x=length).

      Every waveguide will be single-use, so we'll need to create lots of (mostly unique)
    `Pattern`s, and this function will get called very often.
    """
    pat = Pattern()
    pat.rect(layer=layer, xmin=0, xmax=length, yctr=0, ly=width)
    pat.ports = {
        'input': Port(offset=(0, 0), rotation=0, ptype=ptype),
        'output': Port(offset=(length, 0), rotation=pi, ptype=ptype),
        }
    return pat


def map_layer(layer: layer_t) -> layer_t:
    """
    Map from a strings to GDS layer numbers
    """
    layer_mapping = {
        'M1': (10, 0),
        'M2': (20, 0),
        'V1': (30, 0),
        }
    return layer_mapping.get(layer, layer)


#
# Now we can start building up our library (collection of static cells) and pathing tools.
#
#   If any of the operations below are confusing, you can cross-reference against the `RenderPather`
# tutorial, which handles some things more explicitly (e.g. via placement) and simplifies others
# (e.g. geometry definition).
#
def main() -> None:
    # Build some patterns (static cells) using the above functions and store them in a library
    library = Library()
    library['pad'] = make_pad()
    library['m1_bend'] = make_bend(layer='M1', ptype='m1wire', width=M1_WIDTH)
    library['m2_bend'] = make_bend(layer='M2', ptype='m2wire', width=M2_WIDTH)
    library['v1_via'] = make_via(
        layer_top='M2',
        layer_via='V1',
        layer_bot='M1',
        width_top=M2_WIDTH,
        width_via=V1_WIDTH,
        width_bot=M1_WIDTH,
        ptype_bot='m1wire',
        ptype_top='m2wire',
        )

    #
    # Now, define two tools.
    # M1_tool will route on M1, using wires with M1_WIDTH
    # M2_tool will route on M2, using wires with M2_WIDTH
    # Both tools are able to automatically transition from the other wire type (with a via)
    #
    #   Note that while we use BasicTool for this tutorial, you can define your own `Tool`
    # with arbitrary logic inside -- e.g. with single-use bends, complex transition rules,
    # transmission line geometry, or other features.
    #
    M1_tool = BasicTool(
        straight = (
            # First, we need a function which takes in a length and spits out an M1 wire
            lambda length: make_straight_wire(layer='M1', ptype='m1wire', width=M1_WIDTH, length=length),
            'input',    # When we get a pattern from make_straight_wire, use the port named 'input' as the input
            'output',   # and use the port named 'output' as the output
            ),
        bend = (
            library.abstract('m1_bend'),    # When we need a bend, we'll reference the pattern we generated earlier
            'input',    # To orient it clockwise, use the port named 'input' as the input
            'output',   # and 'output' as the output
            ),
        transitions = {  # We can automate transitions for different (normally incompatible) port types
            'm2wire': (  # For example, when we're attaching to a port with type 'm2wire'
                library.abstract('v1_via'),   # we can place a V1 via
                'top',     # using the port named 'top' as the input (i.e. the M2 side of the via)
                'bottom',  # and using the port named 'bottom' as the output
                ),
            },
        default_out_ptype = 'm1wire',   # Unless otherwise requested, we'll default to trying to stay on M1
        )

    M2_tool = BasicTool(
        straight = (
            # Again, we use make_straight_wire, but this time we set parameters for M2
            lambda length: make_straight_wire(layer='M2', ptype='m2wire', width=M2_WIDTH, length=length),
            'input',
            'output',
            ),
        bend = (
            library.abstract('m2_bend'),    # and we use an M2 bend
            'input',
            'output',
            ),
        transitions = {
            'm1wire': (
                library.abstract('v1_via'),     # We still use the same via,
                'bottom',                       # but the input port is now 'bottom'
                'top',                          # and the output port is now 'top'
                ),
            },
        default_out_ptype = 'm2wire',       # We default to trying to stay on M2
        )

    #
    # Create a new pather which writes to `library` and uses `M2_tool` as its default tool.
    # Then, place some pads and start routing wires!
    #
    pather = Pather(library, tools=M2_tool)

    # Place two pads, and define their ports as 'VCC' and 'GND'
    pather.place('pad', offset=(18_000, 30_000), port_map={'wire_port': 'VCC'})
    pather.place('pad', offset=(18_000, 60_000), port_map={'wire_port': 'GND'})
    # Add some labels to make the pads easier to distinguish
    pather.pattern.label(layer='M2', string='VCC', offset=(18e3, 30e3))
    pather.pattern.label(layer='M2', string='GND', offset=(18e3, 60e3))

    # Path VCC forward (in this case south) and turn clockwise 90 degrees (ccw=False)
    # The total distance forward (including the bend's forward component) must be 6um
    pather.path('VCC', ccw=False, length=6_000)

    # Now path VCC to x=0. This time, don't include any bend (ccw=None).
    # Note that if we tried y=0 here, we would get an error since the VCC port is facing in the x-direction.
    pather.path_to('VCC', ccw=None, x=0)

    # Path GND forward by 5um, turning clockwise 90 degrees.
    # This time we use shorthand (bool(0) == False) and omit the parameter labels
    # Note that although ccw=0 is equivalent to ccw=False, ccw=None is not!
    pather.path('GND', 0, 5_000)

    # This time, path GND until it matches the current x-coordinate of VCC. Don't place a bend.
    pather.path_to('GND', None, x=pather['VCC'].offset[0])

    # Now, start using M1_tool for GND.
    # Since we have defined an M2-to-M1 transition for BasicPather, we don't need to place one ourselves.
    #   If we wanted to place our via manually, we could add `pather.plug('m1_via', {'GND': 'top'})` here
    # and achieve the same result without having to define any transitions in M1_tool.
    #   Note that even though we have changed the tool used for GND, the via doesn't get placed until
    # the next time we draw a path on GND (the pather.mpath() statement below).
    pather.retool(M1_tool, keys=['GND'])

    # Bundle together GND and VCC, and path the bundle forward and counterclockwise.
    #   Pick the distance so that the leading/outermost wire (in this case GND) ends up at x=-10_000.
    # Other wires in the bundle (in this case VCC) should be spaced at 5_000 pitch (so VCC ends up at x=-5_000)
    #
    #  Since we recently retooled GND, its path starts with a via down to M1 (included in the distance
    # calculation), and its straight segment and bend will be drawn using M1 while VCC's are drawn with M2.
    pather.mpath(['GND', 'VCC'], ccw=True, xmax=-10_000, spacing=5_000)

    # Now use M1_tool as the default tool for all ports/signals.
    # Since VCC does not have an explicitly assigned tool, it will now transition down to M1.
    pather.retool(M1_tool)

    # Path the GND + VCC bundle forward and counterclockwise by 90 degrees.
    #   The total extension (travel distance along the forward direction) for the longest segment (in
    # this case the segment being added to GND) should be exactly 50um.
    # After turning, the wire pitch should be reduced only 1.2um.
    pather.mpath(['GND', 'VCC'], ccw=True, emax=50_000, spacing=1_200)

    # Make a U-turn with the bundle and expand back out to 4.5um wire pitch.
    #   Here, emin specifies the travel distance for the shortest segment. For the first mpath() call
    # that applies to VCC, and for teh second call, that applies to GND; the relative lengths of the
    # segments depend on their starting positions and their ordering within the bundle.
    pather.mpath(['GND', 'VCC'], ccw=False, emin=1_000, spacing=1_200)
    pather.mpath(['GND', 'VCC'], ccw=False, emin=2_000, spacing=4_500)

    # Now, set the default tool back to M2_tool. Note that GND remains on M1 since it has been
    # explicitly assigned a tool. We could `del pather.tools['GND']` to force it to use the default.
    pather.retool(M2_tool)

    # Now path both ports to x=-28_000.
    #   When ccw is not None, xmin constrains the trailing/innermost port to stop at the target x coordinate,
    # However, with ccw=None, all ports stop at the same coordinate, and so specifying xmin= or xmax= is
    # equivalent.
    pather.mpath(['GND', 'VCC'], None, xmin=-28_000)

    # Further extend VCC out to x=-50_000, and specify that we would like to get an output on M1.
    #  This results in a via at the end of the wire (instead of having one at the start like we got
    # when using pather.retool().
    pather.path_to('VCC', None, -50_000, out_ptype='m1wire')

    # Save the pather's pattern into our library
    library['Pather_and_BasicTool'] = pather.pattern

    # Convert from text-based layers to numeric layers for GDS, and output the file
    library.map_layers(map_layer)
    writefile(library, 'pather.gds', **GDS_OPTS)


if __name__ == '__main__':
    main()
