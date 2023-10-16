"""
Manual wire/waveguide routing (`Pather`)
"""
from typing import Self, Sequence, MutableMapping, Mapping
import copy
import logging
from pprint import pformat

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from ..pattern import Pattern
from ..library import ILibrary, SINGLE_USE_PREFIX
from ..error import PortError, BuildError
from ..ports import PortList, Port
from ..abstract import Abstract
from ..utils import SupportsBool
from .tools import Tool
from .utils import ell
from .builder import Builder


logger = logging.getLogger(__name__)


class Pather(Builder):
    """
      An extension of `Builder` which provides functionality for routing and attaching
    single-use patterns (e.g. wires or waveguides) and bundles / buses of such patterns.

      `Pather` is mostly concerned with calculating how long each wire should be. It calls
    out to `Tool.path` functions provided by subclasses of `Tool` to build the actual patterns.
    `Tool`s are assigned on a per-port basis and stored in `.tools`; a key of `None` represents
    a "default" `Tool` used for all ports which do not have a port-specific `Tool` assigned.


    Examples: Creating a Pather
    ===========================
    - `Pather(library, tools=my_tool)` makes an empty pattern with no ports. The pattern
        is not added into `library` and must later be added with e.g.
        `library['mypat'] = pather.pattern`.
        The default wire/waveguide generating tool for all ports is set to `my_tool`.

    - `Pather(library, ports={'in': Port(...), 'out': ...}, name='mypat', tools=my_tool)`
        makes an empty pattern, adds the given ports, and places it into `library`
        under the name `'mypat'`. The default wire/waveguide generating tool
        for all ports is set to `my_tool`

    - `Pather(..., tools={'in': top_metal_40um, 'out': bottom_metal_1um, None: my_tool})`
        assigns specific tools to individual ports, and `my_tool` as a default for ports
        which are not specified.

    - `Pather.interface(other_pat, port_map=['A', 'B'], library=library, tools=my_tool)`
        makes a new (empty) pattern, copies over ports 'A' and 'B' from
        `other_pat`, and creates additional ports 'in_A' and 'in_B' facing
        in the opposite directions. This can be used to build a device which
        can plug into `other_pat` (using the 'in_*' ports) but which does not
        itself include `other_pat` as a subcomponent.

    - `Pather.interface(other_pather, ...)` does the same thing as
        `Builder.interface(other_builder.pattern, ...)` but also uses
        `other_builder.library` as its library by default.


    Examples: Adding to a pattern
    =============================
    - `pather.path('my_port', ccw=True, distance)` creates a "wire" for which the output
        port is `distance` units away along the axis of `'my_port'` and rotated 90 degrees
        counterclockwise (since `ccw=True`) relative to `'my_port'`. The wire is `plug`ged
        into the existing `'my_port'`, causing the port to move to the wire's output.

        There is no formal guarantee about how far off-axis the output will be located;
        there may be a significant width to the bend that is used to accomplish the 90 degree
        turn. However, an error is raised if `distance` is too small to fit the bend.

    - `pather.path('my_port', ccw=None, distance)` creates a straight wire with a length
        of `distance` and `plug`s it into `'my_port'`.

    - `pather.path_to('my_port', ccw=False, position)` creates a wire which starts at
        `'my_port'` and has its output at the specified `position`, pointing 90 degrees
        clockwise relative to the input. Again, the off-axis position or distance to the
        output is not specified, so `position` takes the form of a single coordinate. To
        ease debugging, position may be specified as `x=position` or `y=position` and an
        error will be raised if the wrong coordinate is given.

    - `pather.mpath(['A', 'B', 'C'], ..., spacing=spacing)` is a superset of `path`
        and `path_to` which can act on multiple ports simultaneously. Each port's wire is
        generated using its own `Tool` (or the default tool if left unspecified).
        The output ports are spaced out by `spacing` along the input ports' axis, unless
        `ccw=None` is specified (i.e. no bends) in which case they all end at the same
        destination coordinate.

    - `pather.plug(wire, {'myport': 'A'})` places port 'A' of `wire` at 'myport'
        of `pather.pattern`. If `wire` has only two ports (e.g. 'A' and 'B'), no `map_out`,
        argument is provided, and the `inherit_name` argument is not explicitly
        set to `False`, the unconnected port of `wire` is automatically renamed to
        'myport'. This allows easy extension of existing ports without changing
        their names or having to provide `map_out` each time `plug` is called.

    - `pather.place(pad, offset=(10, 10), rotation=pi / 2, port_map={'A': 'gnd'})`
        instantiates `pad` at the specified (x, y) offset and with the specified
        rotation, adding its ports to those of `pather.pattern`. Port 'A' of `pad` is
        renamed to 'gnd' so that further routing can use this signal or net name
        rather than the port name on the original `pad` device.

    - `pather.retool(tool)` or `pather.retool(tool, ['in', 'out', None])` can change
        which tool is used for the given ports (or as the default tool). Useful
        when placing vias or using multiple waveguide types along a route.
    """
    __slots__ = ('tools',)

    library: ILibrary
    """
    Library from which existing patterns should be referenced, and to which
    new ones should be added
    """

    tools: dict[str | None, Tool]
    """
    Tool objects are used to dynamically generate new single-use `Pattern`s
    (e.g wires or waveguides) to be plugged into this device. A key of `None`
    indicates the default `Tool`.
    """

    def __init__(
            self,
            library: ILibrary,
            *,
            pattern: Pattern | None = None,
            ports: str | Mapping[str, Port] | None = None,
            tools: Tool | MutableMapping[str | None, Tool] | None = None,
            name: str | None = None,
            ) -> None:
        """
        Args:
            library: The library from which referenced patterns will be taken,
                and where new patterns (e.g. generated by the `tools`) will be placed.
            pattern: The pattern which will be modified by subsequent operations.
                If `None` (default), a new pattern is created.
            ports: Allows specifying the initial set of ports, if `pattern` does
                not already have any ports (or is not provided). May be a string,
                in which case it is interpreted as a name in `library`.
                Default `None` (no ports).
            tools: A mapping of {port: tool} which specifies what `Tool` should be used
                to generate waveguide or wire segments when `path`/`path_to`/`mpath`
                are called. Relies on `Tool.path` implementations.
            name: If specified, `library[name]` is set to `self.pattern`.
        """
        self._dead = False
        self.library = library
        if pattern is not None:
            self.pattern = pattern
        else:
            self.pattern = Pattern()

        if ports is not None:
            if self.pattern.ports:
                raise BuildError('Ports supplied for pattern with pre-existing ports!')
            if isinstance(ports, str):
                ports = library.abstract(ports).ports

            self.pattern.ports.update(copy.deepcopy(dict(ports)))

        if name is not None:
            library[name] = self.pattern

        if tools is None:
            self.tools = {}
        elif isinstance(tools, Tool):
            self.tools = {None: tools}
        else:
            self.tools = dict(tools)

    @classmethod
    def from_builder(
            cls,
            builder: Builder,
            *,
            tools: Tool | MutableMapping[str | None, Tool] | None = None,
            ) -> 'Pather':
        """
        Construct a `Pather` by adding tools to a `Builder`.

        Args:
            builder: Builder to turn into a Pather
            tools: Tools for the `Pather`

        Returns:
            A new Pather object, using `builder.library` and `builder.pattern`.
        """
        new = Pather(library=builder.library, tools=tools, pattern=builder.pattern)
        return new

    @classmethod
    def interface(
            cls,
            source: PortList | Mapping[str, Port] | str,
            *,
            library: ILibrary | None = None,
            tools: Tool | MutableMapping[str | None, Tool] | None = None,
            in_prefix: str = 'in_',
            out_prefix: str = '',
            port_map: dict[str, str] | Sequence[str] | None = None,
            name: str | None = None,
            ) -> 'Pather':
        """
        Wrapper for `Pattern.interface()`, which returns a Pather instead.

        Args:
            source: A collection of ports (e.g. Pattern, Builder, or dict)
                from which to create the interface. May be a pattern name if
                `library` is provided.
            library: Library from which existing patterns should be referenced,
                and to which the new one should be added (if named). If not provided,
                `source.library` must exist and will be used.
            tools: `Tool`s which will be used by the pather for generating new wires
                or waveguides (via `path`/`path_to`/`mpath`).
            in_prefix: Prepended to port names for newly-created ports with
                reversed directions compared to the current device.
            out_prefix: Prepended to port names for ports which are directly
                copied from the current device.
            port_map: Specification for ports to copy into the new device:
                - If `None`, all ports are copied.
                - If a sequence, only the listed ports are copied
                - If a mapping, the listed ports (keys) are copied and
                    renamed (to the values).

        Returns:
            The new pather, with an empty pattern and 2x as many ports as
              listed in port_map.

        Raises:
            `PortError` if `port_map` contains port names not present in the
                current device.
            `PortError` if applying the prefixes results in duplicate port
                names.
        """
        if library is None:
            if hasattr(source, 'library') and isinstance(source.library, ILibrary):
                library = source.library
            else:
                raise BuildError('No library provided (and not present in `source.library`')

        if tools is None and hasattr(source, 'tools') and isinstance(source.tools, dict):
            tools = source.tools

        if isinstance(source, str):
            source = library.abstract(source).ports

        pat = Pattern.interface(source, in_prefix=in_prefix, out_prefix=out_prefix, port_map=port_map)
        new = Pather(library=library, pattern=pat, name=name, tools=tools)
        return new

    def __repr__(self) -> str:
        s = f'<Pather {self.pattern} L({len(self.library)}) {pformat(self.tools)}>'
        return s

    def retool(
            self,
            tool: Tool,
            keys: str | Sequence[str | None] | None = None,
            ) -> Self:
        """
        Update the `Tool` which will be used when generating `Pattern`s for the ports
        given by `keys`.

        Args:
            tool: The new `Tool` to use for the given ports.
            keys: Which ports the tool should apply to. `None` indicates the default tool,
                used when there is no matching entry in `self.tools` for the port in question.

        Returns:
            self
        """
        if keys is None or isinstance(keys, str):
            self.tools[keys] = tool
        else:
            for key in keys:
                self.tools[key] = tool
        return self

    def path(
            self,
            portspec: str,
            ccw: SupportsBool | None,
            length: float,
            *,
            tool_port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Self:
        """
        Create a "wire"/"waveguide" and `plug` it into the port `portspec`, with the aim
        of traveling exactly `length` distance.

        The wire will travel `length` distance along the port's axis, an an unspecified
        (tool-dependent) distance in the perpendicular direction. The output port will
        be rotated (or not) based on the `ccw` parameter.

        Args:
            portspec: The name of the port into which the wire will be plugged.
            ccw: If `None`, the output should be along the same axis as the input.
                Otherwise, cast to bool and turn counterclockwise if True
                and clockwise otherwise.
            length: The total distance from input to output, along the input's axis only.
                (There may be a tool-dependent offset along the other axis.)
            tool_port_names: The names of the ports on the generated pattern. It is unlikely
                that you will need to change these. The first port is the input (to be
                connected to `portspec`).

        Returns:
            self

        Raises:
            BuildError if `distance` is too small to fit the bend (if a bend is present).
            LibraryError if no valid name could be picked for the pattern.
        """
        if self._dead:
            logger.error('Skipping path() since device is dead')
            return self

        tool = self.tools.get(portspec, self.tools[None])
        in_ptype = self.pattern[portspec].ptype
        tree = tool.path(ccw, length, in_ptype=in_ptype, port_names=tool_port_names, **kwargs)
        abstract = self.library << tree
        return self.plug(abstract, {portspec: tool_port_names[0]})

    def path_to(
            self,
            portspec: str,
            ccw: SupportsBool | None,
            position: float | None = None,
            *,
            x: float | None = None,
            y: float | None = None,
            tool_port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Self:
        """
        Create a "wire"/"waveguide" and `plug` it into the port `portspec`, with the aim
        of ending exactly at a target position.

        The wire will travel so that the output port will be placed at exactly the target
        position along the input port's axis. There can be an unspecified (tool-dependent)
        offset in the perpendicular direction. The output port will be rotated (or not)
        based on the `ccw` parameter.

        Args:
            portspec: The name of the port into which the wire will be plugged.
            ccw: If `None`, the output should be along the same axis as the input.
                Otherwise, cast to bool and turn counterclockwise if True
                and clockwise otherwise.
            position: The final port position, along the input's axis only.
                (There may be a tool-dependent offset along the other axis.)
                Only one of `position`, `x`, and `y` may be specified.
            x: The final port position along the x axis.
                `portspec` must refer to a horizontal port if `x` is passed, otherwise a
                BuildError will be raised.
            y: The final port position along the y axis.
                `portspec` must refer to a vertical port if `y` is passed, otherwise a
                BuildError will be raised.
            tool_port_names: The names of the ports on the generated pattern. It is unlikely
                that you will need to change these. The first port is the input (to be
                connected to `portspec`).

        Returns:
            self

        Raises:
            BuildError if `position`, `x`, or `y` is too close to fit the bend (if a bend
                is present).
            BuildError if `x` or `y` is specified but does not match the axis of `portspec`.
            BuildError if more than one of `x`, `y`, and `position` is specified.
        """
        if self._dead:
            logger.error('Skipping path_to() since device is dead')
            return self

        pos_count = sum(vv is not None for vv in (position, x, y))
        if pos_count > 1:
            raise BuildError('Only one of `position`, `x`, and `y` may be specified at once')
        if pos_count < 1:
            raise BuildError('One of `position`, `x`, and `y` must be specified')

        port = self.pattern[portspec]
        if port.rotation is None:
            raise PortError(f'Port {portspec} has no rotation and cannot be used for path_to()')

        if not numpy.isclose(port.rotation % (pi / 2), 0):
            raise BuildError('path_to was asked to route from non-manhattan port')

        is_horizontal = numpy.isclose(port.rotation % pi, 0)
        if is_horizontal:
            if y is not None:
                raise BuildError('Asked to path to y-coordinate, but port is horizontal')
            if position is None:
                position = x
        else:
            if x is not None:
                raise BuildError('Asked to path to x-coordinate, but port is vertical')
            if position is None:
                position = y

        x0, y0 = port.offset
        if is_horizontal:
            if numpy.sign(numpy.cos(port.rotation)) == numpy.sign(position - x0):
                raise BuildError(f'path_to routing to behind source port: x0={x0:g} to {position:g}')
            length = numpy.abs(position - x0)
        else:
            if numpy.sign(numpy.sin(port.rotation)) == numpy.sign(position - y0):
                raise BuildError(f'path_to routing to behind source port: y0={y0:g} to {position:g}')
            length = numpy.abs(position - y0)

        return self.path(portspec, ccw, length, tool_port_names=tool_port_names, **kwargs)

    def mpath(
            self,
            portspec: str | Sequence[str],
            ccw: SupportsBool | None,
            *,
            spacing: float | ArrayLike | None = None,
            set_rotation: float | None = None,
            tool_port_names: tuple[str, str] = ('A', 'B'),
            force_container: bool = False,
            base_name: str = SINGLE_USE_PREFIX + 'mpath',
            **kwargs,
            ) -> Self:
        """
        `mpath` is a superset of `path` and `path_to` which can act on bundles or buses
        of "wires or "waveguides".

        The wires will travel so that the output ports will be placed at well-defined
        locations along the axis of their input ports, but may have arbitrary (tool-
        dependent) offsets in the perpendicular direction.

        If `ccw` is not `None`, the wire bundle will turn 90 degres in either the
        clockwise (`ccw=False`) or counter-clockwise (`ccw=True`) direction. Within the
        bundle, the center-to-center wire spacings after the turn are set by `spacing`,
        which is required when `ccw` is not `None`. The final position of bundle as a
        whole can be set in a number of ways:

             =A>---------------------------V     turn direction: `ccw=False`
                       =B>-------------V   |
         =C>-----------------------V   |
           =D=>----------------V   |
                               |

                               x---x---x---x  `spacing` (can be scalar or array)

                        <-------------->      `emin=`
                        <------>              `bound_type='min_past_furthest', bound=`
          <-------------------------------->  `emax=`
                               x              `pmin=`
                                           x  `pmax=`

            - `emin=`, equivalent to `bound_type='min_extension', bound=`
                The total extension value for the furthest-out port (B in the diagram).
            - `emax=`, equivalent to `bound_type='max_extension', bound=`:
                The total extension value for the closest-in port (C in the diagram).
            - `pmin=`, equivalent to `xmin=`, `ymin=`, or `bound_type='min_position', bound=`:
                The coordinate of the innermost bend (D's bend).
                The x/y versions throw an error if they do not match the port axis (for debug)
            - `pmax=`, `xmax=`, `ymax=`, or `bound_type='max_position', bound=`:
                The coordinate of the outermost bend (A's bend).
                The x/y versions throw an error if they do not match the port axis (for debug)
            - `bound_type='min_past_furthest', bound=`:
                The distance between furthest out-port (B) and the innermost bend (D's bend).

        If `ccw=None`, final output positions (along the input axis) of all wires will be
        identical (i.e. wires will all be cut off evenly). In this case, `spacing=None` is
        required. In this case, `emin=` and `emax=` are equivalent to each other, and
        `pmin=`, `pmax=`, `xmin=`, etc. are also equivalent to each other.


        Args:
            portspec: The names of the ports which are to be routed.
            ccw: If `None`, the outputs should be along the same axis as the inputs.
                Otherwise, cast to bool and turn 90 degrees counterclockwise if `True`
                and clockwise otherwise.
            spacing: Center-to-center distance between output ports along the input port's axis.
                Must be provided if (and only if) `ccw` is not `None`.
            set_rotation: If the provided ports have `rotation=None`, this can be used
                to set a rotation for them.
            tool_port_names: The names of the ports on the generated pattern. It is unlikely
                that you will need to change these. The first port is the input (to be
                connected to `portspec`).
            force_container: If `False` (default), and only a single port is provided, the
                generated wire for that port will be referenced directly, rather than being
                wrapped in an additonal `Pattern`.
            base_name: Name to use for the generated `Pattern`. This will be passed through
                `self.library.get_name()` to get a unique name for each new `Pattern`.

        Returns:
            self

        Raises:
            BuildError if the implied length for any wire is too close to fit the bend
                (if a bend is requested).
            BuildError if `xmin`/`xmax` or `ymin`/`ymax` is specified but does not
                match the axis of `portspec`.
            BuildError if an incorrect bound type or spacing is specified.
        """
        if self._dead:
            logger.error('Skipping mpath() since device is dead')
            return self

        bound_types = set()
        if 'bound_type' in kwargs:
            bound_types.add(kwargs['bound_type'])
            bound = kwargs['bound']
        for bt in ('emin', 'emax', 'pmin', 'pmax', 'xmin', 'xmax', 'ymin', 'ymax', 'min_past_furthest'):
            if bt in kwargs:
                bound_types.add(bt)
                bound = kwargs[bt]

        if not bound_types:
            raise BuildError('No bound type specified for mpath')
        elif len(bound_types) > 1:
            raise BuildError(f'Too many bound types specified for mpath: {bound_types}')
        bound_type = tuple(bound_types)[0]

        if isinstance(portspec, str):
            portspec = [portspec]
        ports = self.pattern[tuple(portspec)]

        extensions = ell(ports, ccw, spacing=spacing, bound=bound, bound_type=bound_type, set_rotation=set_rotation)

        if len(ports) == 1 and not force_container:
            # Not a bus, so having a container just adds noise to the layout
            port_name = tuple(portspec)[0]
            return self.path(port_name, ccw, extensions[port_name], tool_port_names=tool_port_names)
        else:
            bld = Pather.interface(source=ports, library=self.library, tools=self.tools)
            for port_name, length in extensions.items():
                bld.path(port_name, ccw, length, tool_port_names=tool_port_names)
            name = self.library.get_name(base_name)
            self.library[name] = bld.pattern
            return self.plug(Abstract(name, bld.pattern.ports), {sp: 'in_' + sp for sp in ports.keys()})       # TODO safe to use 'in_'?

    # TODO def path_join() and def bus_join()?

    def flatten(self) -> Self:
        """
        Flatten the contained pattern, using the contained library to resolve references.

        Returns:
            self
        """
        self.pattern.flatten(self.library)
        return self

