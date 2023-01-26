from typing import Dict, Iterable, List, Tuple, KeysView, ValuesView
from typing import overload, Union, Optional, TypeVar
import warnings
import traceback
import logging
from collections import Counter
from abc import ABCMeta, abstractmethod

import numpy
from numpy import pi
from numpy.typing import ArrayLike, NDArray

from .traits import PositionableImpl, Rotatable, PivotableImpl, Copyable, Mirrorable
from .utils import rotate_offsets_around
from .error import PortError


logger = logging.getLogger(__name__)


P = TypeVar('P', bound='Port')
PL  = TypeVar('PL',  bound='PortList')
PL2 = TypeVar('PL2', bound='PortList')


class Port(PositionableImpl, Rotatable, PivotableImpl, Copyable, Mirrorable):
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
    __slots__ = (
        'ptype', '_rotation',
        # inherited:
        '_offset',
        )

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
                raise PortError('Rotation must be a scalar')
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


class PortList(metaclass=ABCMeta):
    __slots__ = ()      # Allow subclasses to use __slots__

    @property
    @abstractmethod
    def ports(self) -> Dict[str, Port]:
        """ Uniquely-named ports which can be used to snap to other Device instances"""
        pass

    @ports.setter
    @abstractmethod
    def ports(self, value: Dict[str, Port]) -> None:
        pass

    @overload
    def __getitem__(self, key: str) -> Port:
        pass

    @overload
    def __getitem__(self, key: Union[List[str], Tuple[str, ...], KeysView[str], ValuesView[str]]) -> Dict[str, Port]:
        pass

    def __getitem__(self, key: Union[str, Iterable[str]]) -> Union[Port, Dict[str, Port]]:
        """
        For convenience, ports can be read out using square brackets:
        - `pattern['A'] == Port((0, 0), 0)`
        - ```
          pattern[['A', 'B']] == {
              'A': Port((0, 0), 0),
              'B': Port((0, 0), pi),
              }
          ```
        """
        if isinstance(key, str):
            return self.ports[key]
        else:
            return {k: self.ports[k] for k in key}

    # NOTE: Didn't add keys(), items(), values(), __contains__(), etc.
    # because it's weird on stuff like Pattern that contains other lists
    # and because you can just grab .ports and use that instead

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
                raise PortError(f'Unrenamed ports would be overwritten: {duplicates}')

        renamed = {mapping[k]: self.ports.pop(k) for k in mapping.keys()}
        if None in renamed:
            del renamed[None]

        self.ports.update(renamed)      # type: ignore
        return self

    def add_port_pair(
            self: PL,
            offset: ArrayLike,
            rotation: float = 0.0,
            names: Tuple[str, str] = ('A', 'B'),
            ptype: str = 'unk',
            ) -> PL:
        """
        Add a pair of ports with opposing directions at the specified location.

        Args:
            offset: Location at which to add the ports
            rotation: Orientation of the first port. Radians, counterclockwise.
                Default 0.
            names: Names for the two ports. Default 'A' and 'B'
            ptype: Sets the port type for both ports.

        Returns:
            self
        """
        new_ports = {
            names[0]: Port(offset, rotation=rotation, ptype=ptype),
            names[1]: Port(offset, rotation=rotation + pi, ptype=ptype),
            }
        self.check_ports(names)
        self.ports.update(new_ports)
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
            `PortError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `other_names`.
            `PortError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
        """
        if map_in is None:
            map_in = {}

        if map_out is None:
            map_out = {}

        other = set(other_names)

        missing_inkeys = set(map_in.keys()) - set(self.ports.keys())
        if missing_inkeys:
            raise PortError(f'`map_in` keys not present in device: {missing_inkeys}')

        missing_invals = set(map_in.values()) - other
        if missing_invals:
            raise PortError(f'`map_in` values not present in other device: {missing_invals}')

        missing_outkeys = set(map_out.keys()) - other
        if missing_outkeys:
            raise PortError(f'`map_out` keys not present in other device: {missing_outkeys}')

        orig_remaining = set(self.ports.keys()) - set(map_in.keys())
        other_remaining = other - set(map_out.keys()) - set(map_in.values())
        mapped_vals = set(map_out.values())
        mapped_vals.discard(None)

        conflicts_final = orig_remaining & (other_remaining | mapped_vals)
        if conflicts_final:
            raise PortError(f'Device ports conflict with existing ports: {conflicts_final}')

        conflicts_partial = other_remaining & mapped_vals
        if conflicts_partial:
            raise PortError(f'`map_out` targets conflict with non-mapped outputs: {conflicts_partial}')

        map_out_counts = Counter(map_out.values())
        map_out_counts[None] = 0
        conflicts_out = {k for k, v in map_out_counts.items() if v > 1}
        if conflicts_out:
            raise PortError(f'Duplicate targets in `map_out`: {conflicts_out}')

        return self

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
            msg = 'Ports have conflicting types:\n'
            for nn, (k, v) in enumerate(map_in.items()):
                if type_conflicts[nn]:
                    msg += f'{k} | {s_types[nn]}:{o_types[nn]} | {v}\n'
            msg = ''.join(traceback.format_stack()) + '\n' + msg
            warnings.warn(msg, stacklevel=2)

        rotations = numpy.mod(s_rotations - o_rotations - pi, 2 * pi)
        if not has_rot.any():
            if set_rotation is None:
                PortError('Must provide set_rotation if rotation is indeterminate')
            rotations[:] = set_rotation
        else:
            rotations[~has_rot] = rotations[has_rot][0]

        if not numpy.allclose(rotations[:1], rotations):
            rot_deg = numpy.rad2deg(rotations)
            msg = 'Port orientations do not match:\n'
            for nn, (k, v) in enumerate(map_in.items()):
                msg += f'{k} | {rot_deg[nn]:g} | {v}\n'
            raise PortError(msg)

        pivot = o_offsets[0].copy()
        rotate_offsets_around(o_offsets, pivot, rotations[0])
        translations = s_offsets - o_offsets
        if not numpy.allclose(translations[:1], translations):
            msg = 'Port translations do not match:\n'
            for nn, (k, v) in enumerate(map_in.items()):
                msg += f'{k} | {translations[nn]} | {v}\n'
            raise PortError(msg)

        return translations[0], rotations[0], o_offsets[0]
