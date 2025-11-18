"""
Manual wire/waveguide routing (`Pather`)
"""
from typing import Self
from collections.abc import Sequence, Mapping, MutableMapping
import copy
import logging
from pprint import pformat

from ..pattern import Pattern
from ..library import ILibrary
from ..error import BuildError
from ..ports import PortList, Port
from ..utils import SupportsBool
from .tools import Tool
from .pather_mixin import PatherMixin
from .builder import Builder


logger = logging.getLogger(__name__)


class Pather(Builder, PatherMixin):
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
            cls: type['Pather'],
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
            cls: type['Pather'],
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


    def path(
            self,
            portspec: str,
            ccw: SupportsBool | None,
            length: float,
            *,
            plug_into: str | None = None,
            **kwargs,
            ) -> Self:
        """
        Create a "wire"/"waveguide" and `plug` it into the port `portspec`, with the aim
        of traveling exactly `length` distance.

        The wire will travel `length` distance along the port's axis, and an unspecified
        (tool-dependent) distance in the perpendicular direction. The output port will
        be rotated (or not) based on the `ccw` parameter.

        Args:
            portspec: The name of the port into which the wire will be plugged.
            ccw: If `None`, the output should be along the same axis as the input.
                Otherwise, cast to bool and turn counterclockwise if True
                and clockwise otherwise.
            length: The total distance from input to output, along the input's axis only.
                (There may be a tool-dependent offset along the other axis.)
            plug_into: If not None, attempts to plug the wire's output port into the provided
                port on `self`.

        Returns:
            self

        Raises:
            BuildError if `distance` is too small to fit the bend (if a bend is present).
            LibraryError if no valid name could be picked for the pattern.
        """
        if self._dead:
            logger.error('Skipping path() since device is dead')
            return self

        tool_port_names = ('A', 'B')

        tool = self.tools.get(portspec, self.tools[None])
        in_ptype = self.pattern[portspec].ptype
        tree = tool.path(ccw, length, in_ptype=in_ptype, port_names=tool_port_names, **kwargs)
        tname = self.library << tree
        if plug_into is not None:
            output = {plug_into: tool_port_names[1]}
        else:
            output = {}
        self.plug(tname, {portspec: tool_port_names[0], **output})
        return self

    def pathS(
            self,
            portspec: str,
            length: float,
            jog: float,
            *,
            plug_into: str | None = None,
            **kwargs,
            ) -> Self:
        """
        Create an S-shaped "wire"/"waveguide" and `plug` it into the port `portspec`, with the aim
        of traveling exactly `length` distance with an offset `jog` along the other axis (+ve jog is
        left of direction of travel).

        The output port will have the same orientation as the source port (`portspec`).

        This function attempts to use `tool.planS()`, but falls back to `tool.planL()` if the former
        raises a NotImplementedError.

        Args:
            portspec: The name of the port into which the wire will be plugged.
            jog: Total manhattan distance perpendicular to the direction of travel.
                Positive values are to the left of the direction of travel.
            length: The total manhattan distance from input to output, along the input's axis only.
                (There may be a tool-dependent offset along the other axis.)
            plug_into: If not None, attempts to plug the wire's output port into the provided
                port on `self`.

        Returns:
            self

        Raises:
            BuildError if `distance` is too small to fit the s-bend (for nonzero jog).
            LibraryError if no valid name could be picked for the pattern.
        """
        if self._dead:
            logger.error('Skipping pathS() since device is dead')
            return self

        tool_port_names = ('A', 'B')

        tool = self.tools.get(portspec, self.tools[None])
        in_ptype = self.pattern[portspec].ptype
        try:
            tree = tool.pathS(length, jog, in_ptype=in_ptype, port_names=tool_port_names, **kwargs)
        except NotImplementedError:
            # Fall back to drawing two L-bends
            ccw0 = jog > 0
            kwargs_no_out = (kwargs | {'out_ptype': None})
            t_tree0 = tool.path(    ccw0, length / 2, port_names=tool_port_names, in_ptype=in_ptype, **kwargs_no_out)
            t_pat0 = t_tree0.top_pattern()
            (_, jog0), _ = t_pat0[tool_port_names[0]].measure_travel(t_pat0[tool_port_names[1]])
            t_tree1 = tool.path(not ccw0, jog - jog0, port_names=tool_port_names, in_ptype=t_pat0[tool_port_names[1]].ptype, **kwargs)
            t_pat1 = t_tree1.top_pattern()
            (_, jog1), _ = t_pat1[tool_port_names[0]].measure_travel(t_pat1[tool_port_names[1]])

            self.path(portspec,     ccw0, length - jog1, **kwargs_no_out)
            self.path(portspec, not ccw0, jog    - jog0, **kwargs)
            return self

        tname = self.library << tree
        if plug_into is not None:
            output = {plug_into: tool_port_names[1]}
        else:
            output = {}
        self.plug(tname, {portspec: tool_port_names[0], **output})
        return self

