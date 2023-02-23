from typing import TypeVar, Sequence, MutableMapping, Mapping
import copy
import logging

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from ..pattern import Pattern, NamedPattern
from ..ref import Ref
from ..library import MutableLibrary, Tree
from ..error import PortError, BuildError
from ..ports import PortList, Port
from ..abstract import Abstract
from ..utils import SupportsBool
from .tools import Tool
from .utils import ell


logger = logging.getLogger(__name__)


BB  = TypeVar('BB',  bound='Builder')
PP  = TypeVar('PP',  bound='Pather')


class Builder(PortList):
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
    __slots__ = ('pattern', 'library', '_dead')

    pattern: Pattern
    """ Layout of this device """

    library: MutableLibrary | None
    """
    Library from which existing patterns should be referenced, and to which
    new ones should be added
    """

    _dead: bool
    """ If True, plug()/place() are skipped (for debugging)"""

    @property
    def ports(self) -> dict[str, Port]:
        return self.pattern.ports

    @ports.setter
    def ports(self, value: dict[str, Port]) -> None:
        self.pattern.ports = value

    def __init__(
            self,
            library: MutableLibrary | None = None,
            *,
            pattern: Pattern | None = None,
            ports: str | Mapping[str, Port] | None = None,
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
                if library is None:
                    raise BuildError('Ports given as a string, but `library` was `None`!')
                ports = library.abstract(ports).ports

            self.pattern.ports.update(copy.deepcopy(dict(ports)))

        if name is not None:
            if library is None:
                raise BuildError('Name was supplied, but no library was given!')
            library[name] = self.pattern

    @classmethod
    def interface(
            cls,
            source: PortList | Mapping[str, Port] | str,
            *,
            library: MutableLibrary | None = None,
            in_prefix: str = 'in_',
            out_prefix: str = '',
            port_map: dict[str, str] | Sequence[str] | None = None,
            name: str | None = None,
            ) -> 'Builder':
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
            if hasattr(source, 'library') and isinstance(source.library, MutableLibrary):
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

        ports_in = {f'{in_prefix}{name}': port.deepcopy().rotate(pi)
                    for name, port in mapped_ports.items()}
        ports_out = {f'{out_prefix}{name}': port.deepcopy()
                     for name, port in mapped_ports.items()}

        duplicates = set(ports_out.keys()) & set(ports_in.keys())
        if duplicates:
            raise PortError(f'Duplicate keys after prefixing, try a different prefix: {duplicates}')

        new = Builder(library=library, ports={**ports_in, **ports_out}, name=name)
        return new

    def plug(
            self: BB,
            other: Abstract | str | NamedPattern,
            map_in: dict[str, str],
            map_out: dict[str, str | None] | None = None,
            *,
            mirrored: tuple[bool, bool] = (False, False),
            inherit_name: bool = True,
            set_rotation: bool | None = None,
            ) -> BB:
        """
        Instantiate a device `library[name]` into the current device, connecting
          the ports specified by `map_in` and renaming the unconnected
          ports specified by `map_out`.

        Examples:
        =========
        - `my_device.plug(lib, 'subdevice', {'A': 'C', 'B': 'B'}, map_out={'D': 'myport'})`
            instantiates `lib['subdevice']` into `my_device`, plugging ports 'A' and 'B'
            of `my_device` into ports 'C' and 'B' of `subdevice`. The connected ports
            are removed and any unconnected ports from `subdevice` are added to
            `my_device`. Port 'D' of `subdevice` (unconnected) is renamed to 'myport'.

        - `my_device.plug(lib, 'wire', {'myport': 'A'})` places port 'A' of `lib['wire']`
            at 'myport' of `my_device`.
            If `'wire'` has only two ports (e.g. 'A' and 'B'), no `map_out` argument is
            provided, and the `inherit_name` argument is not explicitly set to `False`,
            the unconnected port of `wire` is automatically renamed to 'myport'. This
            allows easy extension of existing ports without changing their names or
            having to provide `map_out` each time `plug` is called.

        Args:
            other: An `Abstract` describing the device to be instatiated.
            map_in: dict of `{'self_port': 'other_port'}` mappings, specifying
                port connections between the two devices.
            map_out: dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for ports in `other`.
            mirrored: Enables mirroring `other` across the x or y axes prior
                to connecting any ports.
            inherit_name: If `True`, and `map_in` specifies only a single port,
                and `map_out` is `None`, and `other` has only two ports total,
                then automatically renames the output port of `other` to the
                name of the port from `self` that appears in `map_in`. This
                makes it easy to extend a device with simple 2-port devices
                (e.g. wires) without providing `map_out` each time `plug` is
                called. See "Examples" above for more info. Default `True`.
            set_rotation: If the necessary rotation cannot be determined from
                the ports being connected (i.e. all pairs have at least one
                port with `rotation=None`), `set_rotation` must be provided
                to indicate how much `other` should be rotated. Otherwise,
                `set_rotation` must remain `None`.

        Returns:
            self

        Raises:
            `PortError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `other_names`.
            `PortError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
            `PortError` if the specified port mapping is not achieveable (the ports
                do not line up)
        """
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

        self.place(other, offset=translation, rotation=rotation, pivot=pivot,
                   mirrored=mirrored, port_map=map_out, skip_port_check=True)
        return self

    def place(
            self: BB,
            other: Abstract | str | NamedPattern,
            *,
            offset: ArrayLike = (0, 0),
            rotation: float = 0,
            pivot: ArrayLike = (0, 0),
            mirrored: tuple[bool, bool] = (False, False),
            port_map: dict[str, str | None] | None = None,
            skip_port_check: bool = False,
            ) -> BB:
        """
        Instantiate the device `other` into the current device, adding its
          ports to those of the current device (but not connecting any ports).

        Mirroring is applied before rotation; translation (`offset`) is applied last.

        Examples:
        =========
        - `my_device.place(pad, offset=(10, 10), rotation=pi / 2, port_map={'A': 'gnd'})`
            instantiates `pad` at the specified (x, y) offset and with the specified
            rotation, adding its ports to those of `my_device`. Port 'A' of `pad` is
            renamed to 'gnd' so that further routing can use this signal or net name
            rather than the port name on the original `pad` device.

        Args:
            other: An `Abstract` describing the device to be instatiated.
            offset: Offset at which to place the instance. Default (0, 0).
            rotation: Rotation applied to the instance before placement. Default 0.
            pivot: Rotation is applied around this pivot point (default (0, 0)).
                Rotation is applied prior to translation (`offset`).
            mirrored: Whether theinstance should be mirrored across the x and y axes.
                Mirroring is applied before translation and rotation.
            port_map: dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for ports in the instantiated device. New names can be
                `None`, which will delete those ports.
            skip_port_check: Can be used to skip the internal call to `check_ports`,
                in case it has already been performed elsewhere.

        Returns:
            self

        Raises:
            `PortError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `library[name].ports`.
            `PortError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
        """
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

        for name, port in ports.items():
            p = port.deepcopy()
            p.mirror2d(mirrored)
            p.rotate_around(pivot, rotation)
            p.translate(offset)
            self.ports[name] = p

        sp = Ref(other.name, mirrored=mirrored)
        sp.rotate_around(pivot, rotation)
        sp.translate(offset)
        self.pattern.refs.append(sp)
        return self

    def translate(self: BB, offset: ArrayLike) -> BB:
        """
        Translate the pattern and all ports.

        Args:
            offset: (x, y) distance to translate by

        Returns:
            self
        """
        self.pattern.translate_elements(offset)
        for port in self.ports.values():
            port.translate(offset)
        return self

    def rotate_around(self: BB, pivot: ArrayLike, angle: float) -> BB:
        """
        Rotate the pattern and all ports.

        Args:
            angle: angle (radians, counterclockwise) to rotate by
            pivot: location to rotate around

        Returns:
            self
        """
        self.pattern.rotate_around(pivot, angle)
        for port in self.ports.values():
            port.rotate_around(pivot, angle)
        return self

    def mirror(self: BB, axis: int) -> BB:
        """
        Mirror the pattern and all ports across the specified axis.

        Args:
            axis: Axis to mirror across (x=0, y=1)

        Returns:
            self
        """
        self.pattern.mirror(axis)
        for p in self.ports.values():
            p.mirror(axis)
        return self

    def set_dead(self: BB) -> BB:
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
        s = f'<Builder {self.pattern} >'    # TODO maybe show lib and tools? in builder repr?
        return s


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

    library: MutableLibrary
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
            library: MutableLibrary,
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

        if tools is None:
            self.tools = {}
        elif isinstance(tools, Tool):
            self.tools = {None: tools}
        else:
            self.tools = dict(tools)

        if name is not None:
            library[name] = self.pattern

    @classmethod
    def mk(
            cls,
            library: MutableLibrary,
            base_name: str,
            *,
            ports: str | Mapping[str, Port] | None= None,
            tools: Tool | MutableMapping[str | None, Tool] | None = None,
            ) -> tuple['Pather', str]:
        """ Name-and-make combination """
        pat = library.create(base_name)
        pather = Pather(library, pattern=pat, ports=ports, tools=tools)
        return pather, pat.name

    @classmethod
    def from_builder(
            cls,
            builder: Builder,
            *,
            library: MutableLibrary | None = None,
            tools: Tool | MutableMapping[str | None, Tool] | None = None,
            ) -> 'Pather':
        """TODO from_builder docs"""
        library = library if library is not None else builder.library
        if library is None:
            raise BuildError('No library available for Pather!')
        new = Pather(library=library, tools=tools, pattern=builder.pattern)
        return new

    @classmethod
    def interface(
            cls,
            source: PortList | Mapping[str, Port] | str,
            *,
            library: MutableLibrary | None = None,
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
            if hasattr(source, 'library') and isinstance(source.library, MutableLibrary):
                library = source.library
            else:
                raise BuildError('No library provided (and not present in `source.library`')

        if tools is None and hasattr(source, 'tools') and isinstance(source.tools, dict):
            tools = source.tools

        new = Pather.from_builder(
            Builder.interface(
                source=source,
                library=library,
                in_prefix=in_prefix,
                out_prefix=out_prefix,
                port_map=port_map,
                name=name,
                ),
            tools=tools,
            )
        return new

    def __repr__(self) -> str:
        s = f'<Pather {self.pattern} >'    # TODO maybe show lib and tools? in builder repr?
        return s

    def retool(
            self: PP,
            tool: Tool,
            keys: str | Sequence[str | None] | None = None,
            ) -> PP:
        if keys is None or isinstance(keys, str):
            self.tools[keys] = tool
        else:
            for key in keys:
                self.tools[key] = tool
        return self

    def path(
            self: PP,
            portspec: str,
            ccw: SupportsBool | None,
            length: float,
            *,
            tool_port_names: Sequence[str] = ('A', 'B'),
            base_name: str = '_path',
            **kwargs,
            ) -> PP:
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
            self: PP,
            portspec: str,
            ccw: SupportsBool | None,
            position: float,
            *,
            tool_port_names: Sequence[str] = ('A', 'B'),
            base_name: str = '_pathto',
            **kwargs,
            ) -> PP:
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

        return self.path(portspec, ccw, length, tool_port_names=tool_port_names, base_name=base_name, **kwargs)

    def mpath(
            self: PP,
            portspec: str | Sequence[str],
            ccw: SupportsBool | None,
            *,
            spacing: float | ArrayLike | None = None,
            set_rotation: float | None = None,
            tool_port_names: Sequence[str] = ('A', 'B'),
            force_container: bool = False,
            base_name: str = '_mpath',
            **kwargs,
            ) -> PP:
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

    def flatten(self: PP) -> PP:
        """
        Flatten the contained pattern, using the contained library to resolve references.

        Returns:
            self
        """
        self.pattern.flatten(self.library)
        return self

