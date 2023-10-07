from typing import Self, Sequence, Mapping, MutableMapping
import copy
import logging
from collections import defaultdict

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from ..pattern import Pattern
from ..ref import Ref
from ..library import ILibrary, Library
from ..error import PortError, BuildError
from ..ports import PortList, Port
from ..abstract import Abstract
from ..utils import SupportsBool
from .tools import Tool, RenderStep
from .utils import ell
from .builder import Builder


logger = logging.getLogger(__name__)


class RenderPather(PortList):
    __slots__ = ('pattern', 'library', 'paths', 'tools', '_dead', )

    pattern: Pattern
    """ Layout of this device """

    library: ILibrary
    """ Library from which patterns should be referenced """

    _dead: bool
    """ If True, plug()/place() are skipped (for debugging) """

    paths: defaultdict[str, list[RenderStep]]

    tools: dict[str | None, Tool]
    """
    Tool objects are used to dynamically generate new single-use Devices
    (e.g wires or waveguides) to be plugged into this device.
    """

    @property
    def ports(self) -> dict[str, Port]:
        return self.pattern.ports

    @ports.setter
    def ports(self, value: dict[str, Port]) -> None:
        self.pattern.ports = value

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

        """
        self._dead = False
        self.paths = defaultdict(list)
        self.library = library
        if pattern is not None:
            self.pattern = pattern
        else:
            self.pattern = Pattern()

        if ports is not None:
            if self.pattern.ports:
                raise BuildError('Ports supplied for pattern with pre-existing ports!')
            if isinstance(ports, str):
                if library is None:
                    raise BuildError('Ports given as a string, but `library` was `None`!')
                ports = library.abstract(ports).ports

            self.pattern.ports.update(copy.deepcopy(dict(ports)))

        if name is not None:
            if library is None:
                raise BuildError('Name was supplied, but no library was given!')
            library[name] = self.pattern

        if tools is None:
            self.tools = {}
        elif isinstance(tools, Tool):
            self.tools = {None: tools}
        else:
            self.tools = dict(tools)

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
            ) -> 'RenderPather':
        """
        Begin building a new device based on all or some of the ports in the
          source device. Do not include the source device; instead use it
          to define ports (the "interface") for the new device.

        The ports specified by `port_map` (default: all ports) are copied to
          new device, and additional (input) ports are created facing in the
          opposite directions. The specified `in_prefix` and `out_prefix` are
          prepended to the port names to differentiate them.

        By default, the flipped ports are given an 'in_' prefix and unflipped
          ports keep their original names, enabling intuitive construction of
          a device that will "plug into" the current device; the 'in_*' ports
          are used for plugging the devices together while the original port
          names are used for building the new device.

        Another use-case could be to build the new device using the 'in_'
          ports, creating a new device which could be used in place of the
          current device.

        Args:
            source: A collection of ports (e.g. Pattern, Builder, or dict)
                from which to create the interface.
            library: Used for buildin functions; if not passed and the source
            library: Library from which existing patterns should be referenced,
                and to which new ones should be added. If not provided,
                the source's library will be used (if available).
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
            The new builder, with an empty pattern and 2x as many ports as
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
        new = RenderPather(library=library, pattern=pat, name=name, tools=tools)
        return new

    def plug(
            self,
            other: Abstract | str,
            map_in: dict[str, str],
            map_out: dict[str, str | None] | None = None,
            *,
            mirrored: bool = False,
            inherit_name: bool = True,
            set_rotation: bool | None = None,
            append: bool = False,
            ) -> Self:
        if self._dead:
            logger.error('Skipping plug() since device is dead')
            return self

        other_tgt: Pattern | Abstract
        if isinstance(other, str):
            other_tgt = self.library.abstract(other)
        if append and isinstance(other, Abstract):
            other_tgt = self.library[other.name]

        # get rid of plugged ports
        for kk in map_in.keys():
            if kk in self.paths:
                self.paths[kk].append(RenderStep('P', None, self.ports[kk].copy(), self.ports[kk].copy(), None))

        plugged = map_in.values()
        for name, port in other_tgt.ports.items():
            if name in plugged:
                continue
            new_name = map_out.get(name, name) if map_out is not None else name
            if new_name is not None and new_name in self.paths:
                self.paths[new_name].append(RenderStep('P', None, port.copy(), port.copy(), None))

        self.pattern.plug(
            other=other_tgt,
            map_in=map_in,
            map_out=map_out,
            mirrored=mirrored,
            inherit_name=inherit_name,
            set_rotation=set_rotation,
            append=append,
            )

        return self

    def place(
            self,
            other: Abstract | str,
            *,
            offset: ArrayLike = (0, 0),
            rotation: float = 0,
            pivot: ArrayLike = (0, 0),
            mirrored: bool = False,
            port_map: dict[str, str | None] | None = None,
            skip_port_check: bool = False,
            append: bool = False,
            ) -> Self:
        if self._dead:
            logger.error('Skipping place() since device is dead')
            return self

        other_tgt: Pattern | Abstract
        if isinstance(other, str):
            other_tgt = self.library.abstract(other)
        if append and isinstance(other, Abstract):
            other_tgt = self.library[other.name]

        for name, port in other_tgt.ports.items():
            new_name = port_map.get(name, name) if port_map is not None else name
            if new_name is not None and new_name in self.paths:
                self.paths[new_name].append(RenderStep('P', None, port.copy(), port.copy(), None))

        self.pattern.place(
            other=other_tgt,
            offset=offset,
            rotation=rotation,
            pivot=pivot,
            mirrored=mirrored,
            port_map=port_map,
            skip_port_check=skip_port_check,
            append=append,
            )

        return self

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
            **kwargs,
            ) -> Self:
        if self._dead:
            logger.error('Skipping path() since device is dead')
            return self

        port = self.pattern[portspec]
        in_ptype = port.ptype
        port_rot = port.rotation
        assert port_rot is not None         # TODO allow manually setting rotation?

        tool = self.tools.get(portspec, self.tools[None])
        # ask the tool for bend size (fill missing dx or dy), check feasibility, and get out_ptype
        out_port, data = tool.planL(ccw, length, in_ptype=in_ptype, **kwargs)

        # Update port
        out_port.rotate_around((0, 0), pi + port_rot)
        out_port.translate(port.offset)

        step = RenderStep('L', tool, port.copy(), out_port.copy(), data)
        self.paths[portspec].append(step)

        self.pattern.ports[portspec] = out_port.copy()

        return self

    def path_to(
            self,
            portspec: str,
            ccw: SupportsBool | None,
            position: float,
            **kwargs,
            ) -> Self:
        if self._dead:
            logger.error('Skipping path_to() since device is dead')
            return self

        port = self.pattern[portspec]
        x, y = port.offset
        if port.rotation is None:
            raise PortError(f'Port {portspec} has no rotation and cannot be used for path_to()')

        if not numpy.isclose(port.rotation % (pi / 2), 0):
            raise BuildError('path_to was asked to route from non-manhattan port')

        is_horizontal = numpy.isclose(port.rotation % pi, 0)
        if is_horizontal:
            if numpy.sign(numpy.cos(port.rotation)) == numpy.sign(position - x):
                raise BuildError(f'path_to routing to behind source port: x={x:g} to {position:g}')
            length = numpy.abs(position - x)
        else:
            if numpy.sign(numpy.sin(port.rotation)) == numpy.sign(position - y):
                raise BuildError(f'path_to routing to behind source port: y={y:g} to {position:g}')
            length = numpy.abs(position - y)

        return self.path(portspec, ccw, length, **kwargs)

    def mpath(
            self,
            portspec: str | Sequence[str],
            ccw: SupportsBool | None,
            *,
            spacing: float | ArrayLike | None = None,
            set_rotation: float | None = None,
            **kwargs,
            ) -> Self:
        if self._dead:
            logger.error('Skipping mpath() since device is dead')
            return self

        bound_types = set()
        if 'bound_type' in kwargs:
            bound_types.add(kwargs['bound_type'])
            bound = kwargs['bound']
        for bt in ('emin', 'emax', 'pmin', 'pmax', 'min_past_furthest'):
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

        if len(ports) == 1:
            # Not a bus, so having a container just adds noise to the layout
            port_name = tuple(portspec)[0]
            self.path(port_name, ccw, extensions[port_name])
        else:
            for port_name, length in extensions.items():
                self.path(port_name, ccw, length)
        return self

    def render(
            self,
            append: bool = True,
            ) -> Self:
        """
        Generate the geometry which has been planned out with `path`/`path_to`/etc.

        Args:
            append: If `True`, the rendered geometry will be directly appended to
                `self.pattern`. Note that it will not be flattened, so if only one
                layer of hierarchy is eliminated.

        Returns:
            self
        """
        lib = self.library
        tool_port_names = ('A', 'B')
        pat = Pattern()

        def render_batch(portspec: str, batch: list[RenderStep], append: bool) -> None:
            assert batch[0].tool is not None
            name = lib << batch[0].tool.render(batch, port_names=tool_port_names)
            pat.ports[portspec] = batch[0].start_port.copy()
            if append:
                pat.plug(lib[name], {portspec: tool_port_names[0]}, append=append)
                del lib[name]       # NOTE if the rendered pattern has refs, those are now in `pat` but not flattened
            else:
                pat.plug(lib.abstract(name), {portspec: tool_port_names[0]}, append=append)

        for portspec, steps in self.paths.items():
            batch: list[RenderStep] = []
            for step in steps:
                appendable_op = step.opcode in ('L', 'S', 'U')
                same_tool = batch and step.tool == batch[0].tool

                # If we can't continue a batch, render it
                if batch and (not appendable_op or not same_tool):
                    render_batch(portspec, batch, append)
                    batch = []

                # batch is emptied already if we couldn't continue it
                if appendable_op:
                    batch.append(step)

                # Opcodes which break the batch go below this line
                if not appendable_op and portspec in pat.ports:
                    del pat.ports[portspec]

            #If the last batch didn't end yet
            if batch:
                render_batch(portspec, batch, append)

        self.paths.clear()
        pat.ports.clear()
        self.pattern.append(pat)

        return self

    def translate(self, offset: ArrayLike) -> Self:
        """
        Translate the pattern and all ports.

        Args:
            offset: (x, y) distance to translate by

        Returns:
            self
        """
        self.pattern.translate_elements(offset)
        return self

    def rotate_around(self, pivot: ArrayLike, angle: float) -> Self:
        """
        Rotate the pattern and all ports.

        Args:
            angle: angle (radians, counterclockwise) to rotate by
            pivot: location to rotate around

        Returns:
            self
        """
        self.pattern.rotate_around(pivot, angle)
        return self

    def mirror(self, axis: int) -> Self:
        """
        Mirror the pattern and all ports across the specified axis.

        Args:
            axis: Axis to mirror across (x=0, y=1)

        Returns:
            self
        """
        self.pattern.mirror(axis)
        return self

    def set_dead(self) -> Self:
        """
        Disallows further changes through `plug()` or `place()`.
        This is meant for debugging:
        ```
            dev.plug(a, ...)
            dev.set_dead()      # added for debug purposes
            dev.plug(b, ...)    # usually raises an error, but now skipped
            dev.plug(c, ...)    # also skipped
            dev.pattern.visualize()     # shows the device as of the set_dead() call
        ```

        Returns:
            self
        """
        self._dead = True
        return self

    def __repr__(self) -> str:
        s = f'<RenderPather {self.pattern} >'    # TODO maybe show lib and tools? in builder repr?
        return s


