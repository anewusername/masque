"""
Manual wire routing tutorial: RenderPather an PathTool
"""
from typing import Callable
from masque import RenderPather, Library, Pattern, Port, layer_t, map_layers
from masque.builder.tools import PathTool
from masque.file.gdsii import writefile

from basic_shapes import GDS_OPTS
from pather import M1_WIDTH, V1_WIDTH, M2_WIDTH, map_layer, make_pad, make_via


def main() -> None:
    #
    #   To illustrate the advantages of using `RenderPather`, we use `PathTool` instead
    # of `BasicTool`. `PathTool` lacks some sophistication (e.g. no automatic transitions)
    # but when used with `RenderPather`, it can consolidate multiple routing steps into
    # a single `Path` shape.
    #
    #   We'll try to nearly replicate the layout from the `Pather` tutorial; see `pather.py`
    # for more detailed descriptions of the individual pathing steps.
    #

    # First, we make a library and generate some of the same patterns as in the pather tutorial
    library = Library()
    library['pad'] = make_pad()
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

    #   `PathTool` is more limited than `BasicTool`. It only generates one type of shape
    # (`Path`), so it only needs to know what layer to draw on, what width to draw with,
    # and what port type to present.
    M1_ptool = PathTool(layer='M1', width=M1_WIDTH, ptype='m1wire')
    M2_ptool = PathTool(layer='M2', width=M2_WIDTH, ptype='m2wire')
    rpather = RenderPather(tools=M2_ptool, library=library)

    # As in the pather tutorial, we make soem pads and labels...
    rpather.place('pad', offset=(18_000, 30_000), port_map={'wire_port': 'VCC'})
    rpather.place('pad', offset=(18_000, 60_000), port_map={'wire_port': 'GND'})
    rpather.pattern.label(layer='M2', string='VCC', offset=(18e3, 30e3))
    rpather.pattern.label(layer='M2', string='GND', offset=(18e3, 60e3))

    # ...and start routing the signals.
    rpather.path('VCC', ccw=False, length=6_000)
    rpather.path_to('VCC', ccw=None, x=0)
    rpather.path('GND', 0, 5_000)
    rpather.path_to('GND', None, x=rpather['VCC'].offset[0])

    # `PathTool` doesn't know how to transition betwen metal layers, so we have to
    # `plug` the via into the GND wire ourselves.
    rpather.plug('v1_via', {'GND': 'top'})
    rpather.retool(M1_ptool, keys=['GND'])
    rpather.mpath(['GND', 'VCC'], ccw=True, xmax=-10_000, spacing=5_000)

    # Same thing on the VCC wire when it goes down to M1.
    rpather.plug('v1_via', {'VCC': 'top'})
    rpather.retool(M1_ptool)
    rpather.mpath(['GND', 'VCC'], ccw=True, emax=50_000, spacing=1_200)
    rpather.mpath(['GND', 'VCC'], ccw=False, emin=1_000, spacing=1_200)
    rpather.mpath(['GND', 'VCC'], ccw=False, emin=2_000, spacing=4_500)

    # And again when VCC goes back up to M2.
    rpather.plug('v1_via', {'VCC': 'bottom'})
    rpather.retool(M2_ptool)
    rpather.mpath(['GND', 'VCC'], None, xmin=-28_000)

    #   Finally, since PathTool has no conception of transitions, we can't
    # just ask it to transition to an 'm1wire' port at the end of the final VCC segment.
    # Instead, we have to calculate the via size ourselves, and adjust the final position
    # to account for it.
    via_size = abs(
          library['v1_via'].ports['top'].offset[0]
        - library['v1_via'].ports['bottom'].offset[0]
        )
    rpather.path_to('VCC', None, -50_000 + via_size)
    rpather.plug('v1_via', {'VCC': 'top'})

    rpather.render()
    library['RenderPather_and_PathTool'] = rpather.pattern


    # Convert from text-based layers to numeric layers for GDS, and output the file
    library.map_layers(map_layer)
    writefile(library, 'render_pather.gds', **GDS_OPTS)


if __name__ == '__main__':
    main()
