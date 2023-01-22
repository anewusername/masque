from typing import Dict, Iterable, List, Tuple, Union, TypeVar, Any, Iterator, Optional, Sequence
from typing import overload, KeysView, ValuesView
import copy
import warnings
import traceback
import logging
from collections import Counter
from abc import ABCMeta

import numpy
from numpy import pi
from numpy.typing import ArrayLike, NDArray

from ..traits import PositionableImpl, Rotatable, PivotableImpl, Copyable, Mirrorable
from ..pattern import Pattern
from ..ref import Ref
from ..library import MutableLibrary
from ..utils import AutoSlots
from ..error import DeviceError
from ..ports import PortList, Port
from .tools import Tool
from .utils import ell


logger = logging.getLogger(__name__)


B  = TypeVar('B',  bound='Builder')
PR  = TypeVar('PR',  bound='PortsRef')


class PortsRef(PortList):
    __slots__ = ('name', 'ports')

    name: str
    """ Name of the pattern this device references """

    ports: Dict[str, Port]
    """ Uniquely-named ports which can be used to snap instances together"""

    def __init__(
            self,
            name: str,
            ports: Dict[str, Port],
            ) -> None:
        self.name = name
        self.ports = copy.deepcopy(ports)

    def build(self, library: MutableLibrary) -> 'Builder':
        """
        Begin building a new device around an instance of the current device
          (rather than modifying the current device).

        Returns:
            The new `Builder` object.
        """
        pat = Pattern(ports=self.ports)
        pat.ref(self.name)
        new = Builder(library=library, pattern=pat, tools=self.tools)   # TODO should Ref have tools?
        return new

    # TODO do we want to store a Ref instead of just a name? then we can translate/rotate/mirror...

    def __repr__(self) -> str:
        s = f'<PortsRef {self.name} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s


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
    __slots__ = ('pattern', 'library', 'tools', '_dead')

    pattern: Pattern
    """ Layout of this device """

    library: MutableLibrary
    """
    Library from which existing patterns should be referenced, and to which
    new ones should be added
    """

    tools: Dict[Optional[str], Tool]
    """
    Tool objects are used to dynamically generate new single-use Devices
    (e.g wires or waveguides) to be plugged into this device.
    """

    _dead: bool
    """ If True, plug()/place() are skipped (for debugging)"""

    def __init__(
            self,
            library: MutableLibrary,
            pattern: Optional[Pattern] = None,
            *,
            tools: Union[None, Tool, Dict[Optional[str], Tool]] = None,
            ) -> None:
        """
        If `ports` is `None`, two default ports ('A' and 'B') are created.
        Both are placed at (0, 0) and have default `ptype`, but 'A' has rotation 0
          (attached devices will be placed to the left) and 'B' has rotation
          pi (attached devices will be placed to the right).
        """
        self.library = library
        self.pattern = pattern or Pattern()

        ## TODO add_port_pair function to add ports at location with rotation
        #if ports is None:
        #    self.ports = {
        #        'A': Port([0, 0], rotation=0),
        #        'B': Port([0, 0], rotation=pi),
        #        }
        #else:
        #    self.ports = copy.deepcopy(ports)

        if tools is None:
            self.tools = {}
        elif isinstance(tools, Tool):
            self.tools = {None: tools}
        else:
            self.tools = tools

        self._dead = False

    def as_interface(
            self,
            in_prefix: str = 'in_',
            out_prefix: str = '',
            port_map: Optional[Union[Dict[str, str], Sequence[str]]] = None
            ) -> 'Builder':
        new = self.pattern.as_interface(
            library=self.library,
            in_prefix=in_prefix,
            out_prefix=out_prefix,
            port_map=port_map,
            )
        new.tools = self.tools
        return new

    def plug(
            self: B,
            other: PR,
            map_in: Dict[str, str],
            map_out: Optional[Dict[str, Optional[str]]] = None,
            *,
            mirrored: Tuple[bool, bool] = (False, False),
            inherit_name: bool = True,
            set_rotation: Optional[bool] = None,
            ) -> B:
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
            other: A `DeviceRef` describing the device to be instatiated.
            map_in: Dict of `{'self_port': 'other_port'}` mappings, specifying
                port connections between the two devices.
            map_out: Dict of `{'old_name': 'new_name'}` mappings, specifying
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
            `DeviceError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `other_names`.
            `DeviceError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
            `DeviceError` if the specified port mapping is not achieveable (the ports
                do not line up)
        """
        if self._dead:
            logger.error('Skipping plug() since device is dead')
            return self

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
            self: B,
            other: PR,
            *,
            offset: ArrayLike = (0, 0),
            rotation: float = 0,
            pivot: ArrayLike = (0, 0),
            mirrored: Tuple[bool, bool] = (False, False),
            port_map: Optional[Dict[str, Optional[str]]] = None,
            skip_port_check: bool = False,
            ) -> B:
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
            other: A `DeviceRef` describing the device to be instatiated.
            offset: Offset at which to place the instance. Default (0, 0).
            rotation: Rotation applied to the instance before placement. Default 0.
            pivot: Rotation is applied around this pivot point (default (0, 0)).
                Rotation is applied prior to translation (`offset`).
            mirrored: Whether theinstance should be mirrored across the x and y axes.
                Mirroring is applied before translation and rotation.
            port_map: Dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for ports in the instantiated device. New names can be
                `None`, which will delete those ports.
            skip_port_check: Can be used to skip the internal call to `check_ports`,
                in case it has already been performed elsewhere.

        Returns:
            self

        Raises:
            `DeviceError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `library[name].ports`.
            `DeviceError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
        """
        if self._dead:
            logger.error('Skipping place() since device is dead')
            return self

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

    def translate(self: B, offset: ArrayLike) -> B:
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

    def rotate_around(self: B, pivot: ArrayLike, angle: float) -> B:
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

    def mirror(self: B, axis: int) -> B:
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

    def set_dead(self: B) -> B:
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
        s = f'<Builder {self.pattern} >'
        # '['
        # for name, port in self.ports.items():
        #     s += f'\n\t{name}: {port}'
        # s += ']>'
        return s

    def retool(
            self: B,
            tool: Tool,
            keys: Union[Optional[str], Sequence[Optional[str]]] = None,
            ) -> B:
        if keys is None or isinstance(keys, str):
            self.tools[keys] = tool
        else:
            for key in keys:
                self.tools[key] = tool
        return self

    def path(
            self: B,
            portspec: str,
            ccw: Optional[bool],
            length: float,
            *,
            tool_port_names: Sequence[str] = ('A', 'B'),
            base_name: str = '_path_',
            **kwargs,
            ) -> B:
        if self._dead:
            logger.error('Skipping path() since device is dead')
            return self

        tool = self.tools.get(portspec, self.tools[None])
        in_ptype = self.pattern[portspec].ptype
        pat = tool.path(ccw, length, in_ptype=in_ptype, port_names=tool_port_names, **kwargs)
        name = self.library.get_name(base_name)
        self.library._set(name, pat)
        return self.plug(PortsRef(name, pat.ports), {portspec: tool_port_names[0]})

    def path_to(
            self: B,
            portspec: str,
            ccw: Optional[bool],
            position: float,
            *,
            tool_port_names: Sequence[str] = ('A', 'B'),
            base_name: str = '_pathto_',
            **kwargs,
            ) -> B:
        if self._dead:
            logger.error('Skipping path_to() since device is dead')
            return self

        port = self.pattern[portspec]
        x, y = port.offset
        if port.rotation is None:
            raise DeviceError(f'Port {portspec} has no rotation and cannot be used for path_to()')

        if not numpy.isclose(port.rotation % (pi / 2), 0):
            raise DeviceError('path_to was asked to route from non-manhattan port')

        is_horizontal = numpy.isclose(port.rotation % pi, 0)
        if is_horizontal:
            if numpy.sign(numpy.cos(port.rotation)) == numpy.sign(position - x):
                raise DeviceError(f'path_to routing to behind source port: x={x:g} to {position:g}')
            length = numpy.abs(position - x)
        else:
            if numpy.sign(numpy.sin(port.rotation)) == numpy.sign(position - y):
                raise DeviceError(f'path_to routing to behind source port: y={y:g} to {position:g}')
            length = numpy.abs(position - y)

        return self.path(portspec, ccw, length, tool_port_names=tool_port_names, base_name=base_name, **kwargs)

    def mpath(
            self: B,
            portspec: Union[str, Sequence[str]],
            ccw: Optional[bool],
            *,
            spacing: Optional[Union[float, ArrayLike]] = None,
            set_rotation: Optional[float] = None,
            tool_port_names: Sequence[str] = ('A', 'B'),
            force_container: bool = False,
            base_name: str = '_mpath_',
            **kwargs,
            ) -> B:
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
            raise DeviceError('No bound type specified for mpath')
        elif len(bound_types) > 1:
            raise DeviceError(f'Too many bound types specified for mpath: {bound_types}')
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
            bld = ports.as_interface(self.library, tools=self.tools)
            for port_name, length in extensions.items():
                bld.path(port_name, ccw, length, tool_port_names=tool_port_names)
            name = self.library.get_name(base_name)
            self.library._set(name, bld.pattern)
            return self.plug(PortsRef(name, pat.ports), {sp: 'in_' + sp for sp in ports.keys()})       # TODO safe to use 'in_'?

    # TODO def path_join() and def bus_join()?


