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

from ..pattern import Pattern
from ..subpattern import SubPattern
from ..traits import PositionableImpl, Rotatable, PivotableImpl, Copyable, Mirrorable
from ..utils import AutoSlots, rotation_matrix_2d
from ..error import DeviceError
from .tools import Tool
from .utils import ell


logger = logging.getLogger(__name__)


P = TypeVar('P', bound='Port')
PL  = TypeVar('PL',  bound='PortList')
PL2 = TypeVar('PL2', bound='PortList')
D  = TypeVar('D',  bound='Device')
DR  = TypeVar('DR',  bound='DeviceRef')


class Port(PositionableImpl, Rotatable, PivotableImpl, Copyable, Mirrorable, metaclass=AutoSlots):
    """
    A point at which a `Device` can be snapped to another `Device`.

    Each port has an `offset` ((x, y) position) and may also have a
      `rotation` (orientation) and a `ptype` (port type).

    The `rotation` is an angle, in radians, measured counterclockwise
      from the +x axis, pointing inwards into the device which owns the port.
      The rotation may be set to `None`, indicating that any orientation is
      allowed (e.g. for a DC electrical port). It is stored modulo 2pi.

    The `ptype` is an arbitrary string, default of `unk` (unknown).
    """
    __slots__ = ('ptype', '_rotation')

    _rotation: Optional[float]
    """ radians counterclockwise from +x, pointing into device body.
        Can be `None` to signify undirected port """

    ptype: str
    """ Port types must match to be plugged together if both are non-zero """

    def __init__(
            self,
            offset: ArrayLike,
            rotation: Optional[float],
            ptype: str = 'unk',
            ) -> None:
        self.offset = offset
        self.rotation = rotation
        self.ptype = ptype

    @property
    def rotation(self) -> Optional[float]:
        """ Rotation, radians counterclockwise, pointing into device body. Can be None. """
        return self._rotation

    @rotation.setter
    def rotation(self, val: float) -> None:
        if val is None:
            self._rotation = None
        else:
            if not numpy.size(val) == 1:
                raise DeviceError('Rotation must be a scalar')
            self._rotation = val % (2 * pi)

    def get_bounds(self):
        return numpy.vstack((self.offset, self.offset))

    def set_ptype(self: P, ptype: str) -> P:
        """ Chainable setter for `ptype` """
        self.ptype = ptype
        return self

    def mirror(self: P, axis: int) -> P:
        self.offset[1 - axis] *= -1
        if self.rotation is not None:
            self.rotation *= -1
            self.rotation += axis * pi
        return self

    def rotate(self: P, rotation: float) -> P:
        if self.rotation is not None:
            self.rotation += rotation
        return self

    def set_rotation(self: P, rotation: Optional[float]) -> P:
        self.rotation = rotation
        return self

    def __repr__(self) -> str:
        if self.rotation is None:
            rot = 'any'
        else:
            rot = str(numpy.rad2deg(self.rotation))
        return f'<{self.offset}, {rot}, [{self.ptype}]>'


class PortList(Copyable, Mirrorable, metaclass=ABCMeta):
    __slots__ = ('ports',)

    ports: Dict[str, Port]
    """ Uniquely-named ports which can be used to snap to other Device instances"""

    @overload
    def __getitem__(self, key: str) -> Port:
        pass

    @overload
    def __getitem__(self, key: Union[List[str], Tuple[str, ...], KeysView[str], ValuesView[str]]) -> Dict[str, Port]:
        pass

    def __getitem__(self, key: Union[str, Iterable[str]]) -> Union[Port, Dict[str, Port]]:
        """
        For convenience, ports can be read out using square brackets:
        - `device['A'] == Port((0, 0), 0)`
        - `device[['A', 'B']] == {'A': Port((0, 0), 0),
                                  'B': Port((0, 0), pi)}`
        """
        if isinstance(key, str):
            return self.ports[key]
        else:
            return {k: self.ports[k] for k in key}

    def rename_ports(
            self: PL,
            mapping: Dict[str, Optional[str]],
            overwrite: bool = False,
            ) -> PL:
        """
        Renames ports as specified by `mapping`.
        Ports can be explicitly deleted by mapping them to `None`.

        Args:
            mapping: Dict of `{'old_name': 'new_name'}` pairs. Names can be mapped
                to `None` to perform an explicit deletion. `'new_name'` can also
                overwrite an existing non-renamed port to implicitly delete it if
                `overwrite` is set to `True`.
            overwrite: Allows implicit deletion of ports if set to `True`; see `mapping`.

        Returns:
            self
        """
        if not overwrite:
            duplicates = (set(self.ports.keys()) - set(mapping.keys())) & set(mapping.values())
            if duplicates:
                raise DeviceError(f'Unrenamed ports would be overwritten: {duplicates}')

        renamed = {mapping[k]: self.ports.pop(k) for k in mapping.keys()}
        if None in renamed:
            del renamed[None]

        self.ports.update(renamed)      # type: ignore
        return self

    def check_ports(
            self: PL,
            other_names: Iterable[str],
            map_in: Optional[Dict[str, str]] = None,
            map_out: Optional[Dict[str, Optional[str]]] = None,
            ) -> PL:
        """
        Given the provided port mappings, check that:
            - All of the ports specified in the mappings exist
            - There are no duplicate port names after all the mappings are performed

        Args:
            other_names: List of port names being considered for inclusion into
                `self.ports` (before mapping)
            map_in: Dict of `{'self_port': 'other_port'}` mappings, specifying
                port connections between the two devices.
            map_out: Dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for unconnected `other_names` ports.

        Returns:
            self

        Raises:
            `DeviceError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `other_names`.
            `DeviceError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
        """
        if map_in is None:
            map_in = {}

        if map_out is None:
            map_out = {}

        other = set(other_names)

        missing_inkeys = set(map_in.keys()) - set(self.ports.keys())
        if missing_inkeys:
            raise DeviceError(f'`map_in` keys not present in device: {missing_inkeys}')

        missing_invals = set(map_in.values()) - other
        if missing_invals:
            raise DeviceError(f'`map_in` values not present in other device: {missing_invals}')

        missing_outkeys = set(map_out.keys()) - other
        if missing_outkeys:
            raise DeviceError(f'`map_out` keys not present in other device: {missing_outkeys}')

        orig_remaining = set(self.ports.keys()) - set(map_in.keys())
        other_remaining = other - set(map_out.keys()) - set(map_in.values())
        mapped_vals = set(map_out.values())
        mapped_vals.discard(None)

        conflicts_final = orig_remaining & (other_remaining | mapped_vals)
        if conflicts_final:
            raise DeviceError(f'Device ports conflict with existing ports: {conflicts_final}')

        conflicts_partial = other_remaining & mapped_vals
        if conflicts_partial:
            raise DeviceError(f'`map_out` targets conflict with non-mapped outputs: {conflicts_partial}')

        map_out_counts = Counter(map_out.values())
        map_out_counts[None] = 0
        conflicts_out = {k for k, v in map_out_counts.items() if v > 1}
        if conflicts_out:
            raise DeviceError(f'Duplicate targets in `map_out`: {conflicts_out}')

        return self

    def as_interface(
            self,
            in_prefix: str = 'in_',
            out_prefix: str = '',
            port_map: Optional[Union[Dict[str, str], Sequence[str]]] = None
            ) -> 'Device':
        """
        Begin building a new device based on all or some of the ports in the
          current device. Do not include the current device; instead use it
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
            The new device, with an empty pattern and 2x as many ports as
              listed in port_map.

        Raises:
            `DeviceError` if `port_map` contains port names not present in the
                current device.
            `DeviceError` if applying the prefixes results in duplicate port
                names.
        """
        if port_map:
            if isinstance(port_map, dict):
                missing_inkeys = set(port_map.keys()) - set(self.ports.keys())
                orig_ports = {port_map[k]: v for k, v in self.ports.items() if k in port_map}
            else:
                port_set = set(port_map)
                missing_inkeys = port_set - set(self.ports.keys())
                orig_ports = {k: v for k, v in self.ports.items() if k in port_set}

            if missing_inkeys:
                raise DeviceError(f'`port_map` keys not present in device: {missing_inkeys}')
        else:
            orig_ports = self.ports

        ports_in = {f'{in_prefix}{name}': port.deepcopy().rotate(pi)
                    for name, port in orig_ports.items()}
        ports_out = {f'{out_prefix}{name}': port.deepcopy()
                    for name, port in orig_ports.items()}

        duplicates = set(ports_out.keys()) & set(ports_in.keys())
        if duplicates:
            raise DeviceError(f'Duplicate keys after prefixing, try a different prefix: {duplicates}')

        new = Device(ports={**ports_in, **ports_out})
        return new

    def find_transform(
            self: PL,
            other: PL2,
            map_in: Dict[str, str],
            *,
            mirrored: Tuple[bool, bool] = (False, False),
            set_rotation: Optional[bool] = None,
            ) -> Tuple[NDArray[numpy.float64], float, NDArray[numpy.float64]]:
        """
        Given a device `other` and a mapping `map_in` specifying port connections,
          find the transform which will correctly align the specified ports.

        Args:
            other: a device
            map_in: Dict of `{'self_port': 'other_port'}` mappings, specifying
                port connections between the two devices.
            mirrored: Mirrors `other` across the x or y axes prior to
                connecting any ports.
            set_rotation: If the necessary rotation cannot be determined from
                the ports being connected (i.e. all pairs have at least one
                port with `rotation=None`), `set_rotation` must be provided
                to indicate how much `other` should be rotated. Otherwise,
                `set_rotation` must remain `None`.

        Returns:
            - The (x, y) translation (performed last)
            - The rotation (radians, counterclockwise)
            - The (x, y) pivot point for the rotation

            The rotation should be performed before the translation.
        """
        s_ports = self[map_in.keys()]
        o_ports = other[map_in.values()]

        s_offsets = numpy.array([p.offset for p in s_ports.values()])
        o_offsets = numpy.array([p.offset for p in o_ports.values()])
        s_types = [p.ptype for p in s_ports.values()]
        o_types = [p.ptype for p in o_ports.values()]

        s_rotations = numpy.array([p.rotation if p.rotation is not None else 0 for p in s_ports.values()])
        o_rotations = numpy.array([p.rotation if p.rotation is not None else 0 for p in o_ports.values()])
        s_has_rot = numpy.array([p.rotation is not None for p in s_ports.values()], dtype=bool)
        o_has_rot = numpy.array([p.rotation is not None for p in o_ports.values()], dtype=bool)
        has_rot = s_has_rot & o_has_rot

        if mirrored[0]:
            o_offsets[:, 1] *= -1
            o_rotations *= -1
        if mirrored[1]:
            o_offsets[:, 0] *= -1
            o_rotations *= -1
            o_rotations += pi

        type_conflicts = numpy.array([st != ot and st != 'unk' and ot != 'unk'
                                      for st, ot in zip(s_types, o_types)])
        if type_conflicts.any():
            ports = numpy.where(type_conflicts)
            msg = 'Ports have conflicting types:\n'
            for nn, (k, v) in enumerate(map_in.items()):
                if type_conflicts[nn]:
                    msg += f'{k} | {s_types[nn]}:{o_types[nn]} | {v}\n'
            msg = ''.join(traceback.format_stack()) + '\n' + msg
            warnings.warn(msg, stacklevel=2)

        rotations = numpy.mod(s_rotations - o_rotations - pi, 2 * pi)
        if not has_rot.any():
            if set_rotation is None:
                DeviceError('Must provide set_rotation if rotation is indeterminate')
            rotations[:] = set_rotation
        else:
            rotations[~has_rot] = rotations[has_rot][0]

        if not numpy.allclose(rotations[:1], rotations):
            rot_deg = numpy.rad2deg(rotations)
            msg = f'Port orientations do not match:\n'
            for nn, (k, v) in enumerate(map_in.items()):
                msg += f'{k} | {rot_deg[nn]:g} | {v}\n'
            raise DeviceError(msg)

        pivot = o_offsets[0].copy()
        rotate_offsets_around(o_offsets, pivot, rotations[0])
        translations = s_offsets - o_offsets
        if not numpy.allclose(translations[:1], translations):
            msg = f'Port translations do not match:\n'
            for nn, (k, v) in enumerate(map_in.items()):
                msg += f'{k} | {translations[nn]} | {v}\n'
            raise DeviceError(msg)

        return translations[0], rotations[0], o_offsets[0]


class DeviceRef(PortList):
    __slots__ = ('name',)

    name: str
    """ Name of the pattern this device references """

    ports: Dict[str, Port]
    """ Uniquely-named ports which can be used to snap to other Device instances"""

    def __init__(
            self,
            name: str,
            ports: Dict[str, Port],
            ) -> None:
        self.name = name
        self.ports = copy.deepcopy(ports)

    def build(self) -> 'Device':
        """
        Begin building a new device around an instance of the current device
          (rather than modifying the current device).

        Returns:
            The new `Device` object.
        """
        pat = Pattern()
        pat.addsp(self.name)
        new = Device(pat, ports=self.ports, tools=self.tools)   # TODO should DeviceRef have tools?
        return new

    # TODO do we want to store a SubPattern instead of just a name? then we can translate/rotate/mirror...

    def __repr__(self) -> str:
        s = f'<DeviceRef {self.name} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s


class Device(PortList):
    """
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
    __slots__ = ('pattern', 'tools', '_dead')

    pattern: Pattern
    """ Layout of this device """

    ports: Dict[str, Port]
    """ Uniquely-named ports which can be used to snap to other Device instances"""

    tools: Dict[Optional[str], Tool]
    """
    Tool objects are used to dynamically generate new single-use Devices
    (e.g wires or waveguides) to be plugged into this device.
    """

    _dead: bool
    """ If True, plug()/place() are skipped (for debugging)"""

    def __init__(
            self,
            pattern: Optional[Pattern] = None,
            ports: Optional[Dict[str, Port]] = None,
            *,
            tools: Union[None, Tool, Dict[Optional[str], Tool]] = None,
            ) -> None:
        """
        If `ports` is `None`, two default ports ('A' and 'B') are created.
        Both are placed at (0, 0) and have default `ptype`, but 'A' has rotation 0
          (attached devices will be placed to the left) and 'B' has rotation
          pi (attached devices will be placed to the right).
        """
        self.pattern = pattern or Pattern()

        if ports is None:
            self.ports = {
                'A': Port([0, 0], rotation=0),
                'B': Port([0, 0], rotation=pi),
                }
        else:
            self.ports = copy.deepcopy(ports)

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
            ) -> 'Device':
        new = PortList.as_interface(
            self,
            in_prefix=in_prefix,
            out_prefix=out_prefix,
            port_map=port_map,
            )
        new.tools = self.tools
        return new

    def plug(
            self: D,
            other: DR,
            map_in: Dict[str, str],
            map_out: Optional[Dict[str, Optional[str]]] = None,
            *,
            mirrored: Tuple[bool, bool] = (False, False),
            inherit_name: bool = True,
            set_rotation: Optional[bool] = None,
            ) -> D:
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
        translation, rotation, pivot = self.find_transform(other, map_in, mirrored=mirrored,
                                                           set_rotation=set_rotation)

        # get rid of plugged ports
        for ki, vi in map_in.items():
            del self.ports[ki]
            map_out[vi] = None

        self.place(other, offset=translation, rotation=rotation, pivot=pivot,
                   mirrored=mirrored, port_map=map_out, skip_port_check=True)
        return self

    def place(
            self: D,
            other: DR,
            *,
            offset: ArrayLike = (0, 0),
            rotation: float = 0,
            pivot: ArrayLike = (0, 0),
            mirrored: Tuple[bool, bool] = (False, False),
            port_map: Optional[Dict[str, Optional[str]]] = None,
            skip_port_check: bool = False,
            ) -> D:
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

        sp = SubPattern(other.name, mirrored=mirrored)
        sp.rotate_around(pivot, rotation)
        sp.translate(offset)
        self.pattern.subpatterns.append(sp)
        return self

    def translate(self: D, offset: ArrayLike) -> D:
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

    def rotate_around(self: D, pivot: ArrayLike, angle: float) -> D:
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

    def mirror(self: D, axis: int) -> D:
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

    def set_dead(self: D) -> D:
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
        s = f'<Device {self.pattern} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s

    def retool(
            self: D,
            tool: Tool,
            keys: Union[Optional[str], Sequence[Optional[str]]] = None,
            ) -> D:
        if keys is None or isinstance(keys, str):
            self.tools[keys] = tool
        else:
            for key in keys:
                self.tools[key] = tool
        return self

    def path(
            self: D,
            portspec: str,
            ccw: Optional[bool],
            length: float,
            *,
            tool_port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> D:
        if self._dead:
            logger.error('Skipping path() since device is dead')
            return self

        tool = self.tools.get(portspec, self.tools[None])
        in_ptype = self.ports[portspec].ptype
        dev = tool.path(ccw, length, in_ptype=in_ptype, port_names=tool_port_names, **kwargs)
        return self.plug(dev, {portspec: tool_port_names[0]})

    def path_to(
            self: D,
            portspec: str,
            ccw: Optional[bool],
            position: float,
            *,
            tool_port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> D:
        if self._dead:
            logger.error('Skipping path_to() since device is dead')
            return self

        port = self.ports[portspec]
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

        return self.path(portspec, ccw, length, tool_port_names=tool_port_names, **kwargs)

    def mpath(
            self: D,
            portspec: Union[str, Sequence[str]],
            ccw: Optional[bool],
            *,
            spacing: Optional[Union[float, ArrayLike]] = None,
            set_rotation: Optional[float] = None,
            tool_port_names: Sequence[str] = ('A', 'B'),
            force_container: bool = False,
            **kwargs,
            ) -> D:
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
        ports = self[tuple(portspec)]

        extensions = ell(ports, ccw, spacing=spacing, bound=bound, bound_type=bound_type, set_rotation=set_rotation)

        if len(ports) == 1 and not force_container:
            # Not a bus, so having a container just adds noise to the layout
            port_name = tuple(portspec)[0]
            return self.path(port_name, ccw, extensions[port_name], tool_port_names=tool_port_names)
        else:
            dev = Device(name='', ports=ports, tools=self.tools).as_interface()
            for name, length in extensions.items():
                dev.path(name, ccw, length, tool_port_names=tool_port_names)
            return self.plug(dev, {sp: 'in_' + sp for sp in ports.keys()})       # TODO safe to use 'in_'?

    # TODO def path_join() and def bus_join()?


def rotate_offsets_around(
        offsets: NDArray[numpy.float64],
        pivot: NDArray[numpy.float64],
        angle: float,
        ) -> NDArray[numpy.float64]:
    offsets -= pivot
    offsets[:] = (rotation_matrix_2d(angle) @ offsets.T).T
    offsets += pivot
    return offsets
