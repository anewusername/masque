from typing import Self, Sequence, MutableMapping, Mapping
import copy
import logging
from pprint import pformat

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from ..pattern import Pattern
from ..library import ILibrary
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
    TODO DOCUMENT Builder
    A `Device` is a combination of a `Pattern` with a set of named `Port`s
      which can be used to "snap" devices together to make complex layouts.

    `Device`s can be as simple as one or two ports (e.g. an electrical pad
    or wire), but can also be used to build and represent a large routed
    layout (e.g. a logical block with multiple I/O connections or even a
    full chip).

    For convenience, ports can be read out using square brackets:
    - `device['A'] == Port((0, 0), 0)`
    - `device[['A', 'B']] == {'A': Port((0, 0), 0), 'B': Port((0, 0), pi)}`

    Examples: Creating a Device
    ===========================
    - `Device(pattern, ports={'A': port_a, 'C': port_c})` uses an existing
        pattern and defines some ports.

    - `Device(ports=None)` makes a new empty pattern with
        default ports ('A' and 'B', in opposite directions, at (0, 0)).

    - `my_device.build('my_layout')` makes a new pattern and instantiates
        `my_device` in it with offset (0, 0) as a base for further building.

    - `my_device.as_interface('my_component', port_map=['A', 'B'])` makes a new
        (empty) pattern, copies over ports 'A' and 'B' from `my_device`, and
        creates additional ports 'in_A' and 'in_B' facing in the opposite
        directions. This can be used to build a device which can plug into
        `my_device` (using the 'in_*' ports) but which does not itself include
        `my_device` as a subcomponent.

    Examples: Adding to a Device
    ============================
    - `my_device.plug(subdevice, {'A': 'C', 'B': 'B'}, map_out={'D': 'myport'})`
        instantiates `subdevice` into `my_device`, plugging ports 'A' and 'B'
        of `my_device` into ports 'C' and 'B' of `subdevice`. The connected ports
        are removed and any unconnected ports from `subdevice` are added to
        `my_device`. Port 'D' of `subdevice` (unconnected) is renamed to 'myport'.

    - `my_device.plug(wire, {'myport': 'A'})` places port 'A' of `wire` at 'myport'
        of `my_device`. If `wire` has only two ports (e.g. 'A' and 'B'), no `map_out`,
        argument is provided, and the `inherit_name` argument is not explicitly
        set to `False`, the unconnected port of `wire` is automatically renamed to
        'myport'. This allows easy extension of existing ports without changing
        their names or having to provide `map_out` each time `plug` is called.

    - `my_device.place(pad, offset=(10, 10), rotation=pi / 2, port_map={'A': 'gnd'})`
        instantiates `pad` at the specified (x, y) offset and with the specified
        rotation, adding its ports to those of `my_device`. Port 'A' of `pad` is
        renamed to 'gnd' so that further routing can use this signal or net name
        rather than the port name on the original `pad` device.
    """
    __slots__ = ('tools',)

    library: ILibrary
    """
    Library from which existing patterns should be referenced, and to which
    new ones should be added
    """

    tools: dict[str | None, Tool]
    """
    Tool objects are used to dynamically generate new single-use Devices
    (e.g wires or waveguides) to be plugged into this device.
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
        # TODO documentation for Builder() constructor

        # TODO MOVE THE BELOW DOCS to PortList
        # If `ports` is `None`, two default ports ('A' and 'B') are created.
        # Both are placed at (0, 0) and have default `ptype`, but 'A' has rotation 0
        #   (attached devices will be placed to the left) and 'B' has rotation
        #   pi (attached devices will be placed to the right).
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
        TODO doc pather.interface
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
            base_name: str = '_path',
            **kwargs,
            ) -> Self:
        if self._dead:
            logger.error('Skipping path() since device is dead')
            return self

        tool = self.tools.get(portspec, self.tools[None])
        in_ptype = self.pattern[portspec].ptype
        pat = tool.path(ccw, length, in_ptype=in_ptype, port_names=tool_port_names, **kwargs)
        name = self.library.get_name(base_name)
        self.library[name] = pat
        return self.plug(Abstract(name, pat.ports), {portspec: tool_port_names[0]})

    def path_to(
            self,
            portspec: str,
            ccw: SupportsBool | None,
            position: float | None = None,
            *,
            x: float | None = None,
            y: float | None = None,
            tool_port_names: tuple[str, str] = ('A', 'B'),
            base_name: str = '_pathto',
            **kwargs,
            ) -> Self:
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
                raise BuildError(f'Asked to path to y-coordinate, but port is horizontal')
            if position is None:
                position = x
        else:
            if x is not None:
                raise BuildError(f'Asked to path to x-coordinate, but port is vertical')
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

        return self.path(portspec, ccw, length, tool_port_names=tool_port_names, base_name=base_name, **kwargs)

    def mpath(
            self,
            portspec: str | Sequence[str],
            ccw: SupportsBool | None,
            *,
            spacing: float | ArrayLike | None = None,
            set_rotation: float | None = None,
            tool_port_names: tuple[str, str] = ('A', 'B'),
            force_container: bool = False,
            base_name: str = '_mpath',
            **kwargs,
            ) -> Self:
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

