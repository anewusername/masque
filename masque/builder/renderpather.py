from typing import Self, Sequence, Mapping, Final
import copy
import logging
from collections import defaultdict

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from ..pattern import Pattern
from ..ref import Ref
from ..library import ILibrary
from ..error import PortError, BuildError
from ..ports import PortList, Port
from ..abstract import Abstract
from ..utils import rotation_matrix_2d
from ..utils import SupportsBool
from .tools import Tool, render_step_t
from .utils import ell
from .builder import Builder


logger = logging.getLogger(__name__)


class RenderPather(PortList):
    __slots__ = ('pattern', 'library', 'paths', 'tools', '_dead', )

    pattern: Pattern
    """ Layout of this device """

    library: ILibrary | None
    """ Library from which patterns should be referenced """

    _dead: bool
    """ If True, plug()/place() are skipped (for debugging) """

    paths: defaultdict[str, list[render_step_t]]
#       op, start_port, dx, dy, o_ptype tool

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
            library: ILibrary | None = None,
            *,
            pattern: Pattern | None = None,
            ports: str | Mapping[str, Port] | None = None,
            name: str | None = None,
            ) -> None:
        """
        # TODO documentation for Builder() constructor

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
                if library is None:
                    raise BuildError('Ports given as a string, but `library` was `None`!')
                ports = library.abstract(ports).ports

            self.pattern.ports.update(copy.deepcopy(dict(ports)))

        if name is not None:
            if library is None:
                raise BuildError('Name was supplied, but no library was given!')
            library[name] = self.pattern

        self.paths = defaultdict(list)

    @classmethod
    def interface(
            cls,
            source: PortList | Mapping[str, Port] | str,
            *,
            library: ILibrary | None = None,
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

        if isinstance(source, str):
            if library is None:
                raise BuildError('Source given as a string, but `library` was `None`!')
            orig_ports = library.abstract(source).ports
        elif isinstance(source, PortList):
            orig_ports = source.ports
        elif isinstance(source, dict):
            orig_ports = source
        else:
            raise BuildError(f'Unable to get ports from {type(source)}: {source}')

        if port_map:
            if isinstance(port_map, dict):
                missing_inkeys = set(port_map.keys()) - set(orig_ports.keys())
                mapped_ports = {port_map[k]: v for k, v in orig_ports.items() if k in port_map}
            else:
                port_set = set(port_map)
                missing_inkeys = port_set - set(orig_ports.keys())
                mapped_ports = {k: v for k, v in orig_ports.items() if k in port_set}

            if missing_inkeys:
                raise PortError(f'`port_map` keys not present in source: {missing_inkeys}')
        else:
            mapped_ports = orig_ports

        ports_in = {f'{in_prefix}{pname}': port.deepcopy().rotate(pi)
                    for pname, port in mapped_ports.items()}
        ports_out = {f'{out_prefix}{pname}': port.deepcopy()
                     for pname, port in mapped_ports.items()}

        duplicates = set(ports_out.keys()) & set(ports_in.keys())
        if duplicates:
            raise PortError(f'Duplicate keys after prefixing, try a different prefix: {duplicates}')

        new = RenderPather(library=library, ports={**ports_in, **ports_out}, name=name)
        return new

    def plug(
            self,
            other: Abstract | str,
            map_in: dict[str, str],
            map_out: dict[str, str | None] | None = None,
            *,
            mirrored: tuple[bool, bool] = (False, False),
            inherit_name: bool = True,
            set_rotation: bool | None = None,
            ) -> Self:
        if self._dead:
            logger.error('Skipping plug() since device is dead')
            return self

        if isinstance(other, str):
            if self.library is None:
                raise BuildError('No library available, but `other` was a string!')
            other = self.library.abstract(other)

        # If asked to inherit a name, check that all conditions are met
        if (inherit_name
                and not map_out
                and len(map_in) == 1
                and len(other.ports) == 2):
            out_port_name = next(iter(set(other.ports.keys()) - set(map_in.values())))
            map_out = {out_port_name: next(iter(map_in.keys()))}

        if map_out is None:
            map_out = {}
        map_out = copy.deepcopy(map_out)

        self.check_ports(other.ports.keys(), map_in, map_out)
        translation, rotation, pivot = self.find_transform(
            other,
            map_in,
            mirrored=mirrored,
            set_rotation=set_rotation,
            )

        # get rid of plugged ports
        for ki, vi in map_in.items():
            del self.ports[ki]
            map_out[vi] = None
            if ki in self.paths:
                self.paths[ki].append(('P', None, 0.0, 0.0, 'unk', None))

        self.place(other, offset=translation, rotation=rotation, pivot=pivot,
                   mirrored=mirrored, port_map=map_out, skip_port_check=True)
        return self

    def place(
            self,
            other: Abstract | str,
            *,
            offset: ArrayLike = (0, 0),
            rotation: float = 0,
            pivot: ArrayLike = (0, 0),
            mirrored: tuple[bool, bool] = (False, False),
            port_map: dict[str, str | None] | None = None,
            skip_port_check: bool = False,
            ) -> Self:
        if self._dead:
            logger.error('Skipping place() since device is dead')
            return self

        if isinstance(other, str):
            if self.library is None:
                raise BuildError('No library available, but `other` was a string!')
            other = self.library.abstract(other)

        if port_map is None:
            port_map = {}

        if not skip_port_check:
            self.check_ports(other.ports.keys(), map_in=None, map_out=port_map)

        ports = {}
        for name, port in other.ports.items():
            new_name = port_map.get(name, name)
            if new_name is None:
                continue
            ports[new_name] = port
            if new_name in self.paths:
                self.paths[new_name].append(('P', None, 0.0, 0.0, 'unk', None))

        for name, port in ports.items():
            p = port.deepcopy()
            p.mirror2d(mirrored)
            p.rotate_around(pivot, rotation)
            p.translate(offset)
            self.ports[name] = p

        ref = Ref(mirrored=mirrored)
        ref.rotate_around(pivot, rotation)
        ref.translate(offset)
        self.pattern.refs[other.name].append(ref)
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
        bend_radius, out_ptype = tool.planL(ccw, length, in_ptype=in_ptype, **kwargs)

        if ccw is None:
            bend_run = 0.0
        elif bool(ccw):
            bend_run = bend_radius
        else:
            bend_run = -bend_radius

        dx, dy = rotation_matrix_2d(port_rot + pi) @ [length, bend_run]

        step: Final = ('L', port.deepcopy(), dx, dy, out_ptype, tool)
        self.paths[portspec].append(step)

        # Update port
        port.offset += (dx, dy)
        if ccw is not None:
            port.rotate((-1 if ccw else 1) * pi / 2)
        port.ptype = out_ptype

        return self

        '''
        - record ('path', port, dx, dy, out_ptype, tool)
        - to render, ccw = {0: None, 1: True, -1: False}[numpy.sign(dx) * numpy.sign(dy) * (-1 if x_start else 1)
            - length is just dx or dy
            - in_ptype and out_ptype are taken directly
        - for sbend: dx and dy are maybe reordered (length and jog)
        '''

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

    def render(self, lib: ILibrary | None = None) -> Self:
        lib = lib if lib is not None else self.library
        assert lib is not None

        tool_port_names = ('A', 'B')
        bb = Builder(lib)

        for portspec, steps in self.paths.items():
            batch: list[render_step_t] = []
            for step in steps:
                opcode, _start_port, _dx, _dy, _out_ptype, tool = step

                appendable_op = opcode in ('L', 'S', 'U')
                same_tool = batch and tool == batch[-1]

                if batch and (not appendable_op or not same_tool):
                    # If we can't continue a batch, render it
                    assert tool is not None
                    assert batch[0][1] is not None
                    name = lib << tool.render(batch, portnames=tool_port_names)
                    bb.ports[portspec] = batch[0][1]
                    bb.plug(name, {portspec: tool_port_names[0]})
                    batch = []

                # batch is emptied already if we couldn't
                if appendable_op:
                    batch.append(step)

                # Opcodes which break the batch go below this line
                if not appendable_op:
                    del bb.ports[portspec]

            if batch:
                # A batch didn't end yet
                assert tool is not None
                assert batch[0][1] is not None
                name = lib << tool.render(batch, portnames=tool_port_names)
                bb.ports[portspec] = batch[0][1]
                bb.plug(name, {portspec: tool_port_names[0]})

        bb.ports.clear()
        self.pattern.append(bb.pattern)

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


