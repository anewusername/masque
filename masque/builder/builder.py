"""
Simplified Pattern assembly (`Builder`)
"""
from typing import Self, Sequence, Mapping
import copy
import logging

from numpy.typing import ArrayLike

from ..pattern import Pattern
from ..library import ILibrary
from ..error import BuildError
from ..ports import PortList, Port
from ..abstract import Abstract


logger = logging.getLogger(__name__)


class Builder(PortList):
    """
      A `Builder` is a helper object used for snapping together multiple
    lower-level patterns at their `Port`s.

      The `Builder` mostly just holds context, in the form of a `Library`,
    in addition to its underlying pattern. This simplifies some calls
    to `plug` and `place`, by making the library implicit.

    `Builder` can also be `set_dead()`, at which point further calls to `plug()`
    and `place()` are ignored (intended for debugging).


    Examples: Creating a Builder
    ===========================
    - `Builder(library, ports={'A': port_a, 'C': port_c}, name='mypat')` makes
        an empty pattern, adds the given ports, and places it into `library`
        under the name `'mypat'`.

    - `Builder(library)` makes an empty pattern with no ports. The pattern
        is not added into `library` and must later be added with e.g.
        `library['mypat'] = builder.pattern`

    - `Builder(library, pattern=pattern, name='mypat')` uses an existing
        pattern (including its ports) and sets `library['mypat'] = pattern`.

    - `Builder.interface(other_pat, port_map=['A', 'B'], library=library)`
        makes a new (empty) pattern, copies over ports 'A' and 'B' from
        `other_pat`, and creates additional ports 'in_A' and 'in_B' facing
        in the opposite directions. This can be used to build a device which
        can plug into `other_pat` (using the 'in_*' ports) but which does not
        itself include `other_pat` as a subcomponent.

    - `Builder.interface(other_builder, ...)` does the same thing as
        `Builder.interface(other_builder.pattern, ...)` but also uses
        `other_builder.library` as its library by default.


    Examples: Adding to a pattern
    =============================
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

    library: ILibrary
    """
    Library from which patterns should be referenced
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
            library: ILibrary,
            *,
            pattern: Pattern | None = None,
            ports: str | Mapping[str, Port] | None = None,
            name: str | None = None,
            ) -> None:
        """
        Args:
            library: The library from which referenced patterns will be taken
            pattern: The pattern which will be modified by subsequent operations.
                If `None` (default), a new pattern is created.
            ports: Allows specifying the initial set of ports, if `pattern` does
                not already have any ports (or is not provided). May be a string,
                in which case it is interpreted as a name in `library`.
                Default `None` (no ports).
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
            ) -> 'Builder':
        """
        Wrapper for `Pattern.interface()`, which returns a Builder instead.

        Args:
            source: A collection of ports (e.g. Pattern, Builder, or dict)
                from which to create the interface. May be a pattern name if
                `library` is provided.
            library: Library from which existing patterns should be referenced,
                and to which the new one should be added (if named). If not provided,
                `source.library` must exist and will be used.
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
                raise BuildError('No library was given, and `source.library` does not have one either.')

        if isinstance(source, str):
            source = library.abstract(source).ports

        pat = Pattern.interface(source, in_prefix=in_prefix, out_prefix=out_prefix, port_map=port_map)
        new = Builder(library=library, pattern=pat, name=name)
        return new

    def plug(
            self,
            other: Abstract | str | Pattern,
            map_in: dict[str, str],
            map_out: dict[str, str | None] | None = None,
            *,
            mirrored: bool = False,
            inherit_name: bool = True,
            set_rotation: bool | None = None,
            append: bool = False,
            ) -> Self:
        """
        Wrapper around `Pattern.plug` which allows a string for `other`.
        The `Builder`'s library is used to dereference the string (or `Abstract`, if
        one is passed with `append=True`).

        Args:
            other: An `Abstract`, string, or `Pattern` describing the device to be instatiated.
            map_in: dict of `{'self_port': 'other_port'}` mappings, specifying
                port connections between the two devices.
            map_out: dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for ports in `other`.
            mirrored: Enables mirroring `other` across the x axis prior to
                connecting any ports.
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
            append: If `True`, `other` is appended instead of being referenced.
                Note that this does not flatten  `other`, so its refs will still
                be refs (now inside `self`).

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
            other = self.library.abstract(other)
        if append and isinstance(other, Abstract):
            other = self.library[other.name]

        self.pattern.plug(
            other=other,
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
            other: Abstract | str | Pattern,
            *,
            offset: ArrayLike = (0, 0),
            rotation: float = 0,
            pivot: ArrayLike = (0, 0),
            mirrored: bool = False,
            port_map: dict[str, str | None] | None = None,
            skip_port_check: bool = False,
            append: bool = False,
            ) -> Self:
        """
        Wrapper around `Pattern.place` which allows a string for `other`.
        The `Builder`'s library is used to dereference the string (or `Abstract`, if
        one is passed with `append=True`).

        Args:
            other: An `Abstract`, string, or `Pattern` describing the device to be instatiated.
            offset: Offset at which to place the instance. Default (0, 0).
            rotation: Rotation applied to the instance before placement. Default 0.
            pivot: Rotation is applied around this pivot point (default (0, 0)).
                Rotation is applied prior to translation (`offset`).
            mirrored: Whether theinstance should be mirrored across the x axis.
                Mirroring is applied before translation and rotation.
            port_map: dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for ports in the instantiated device. New names can be
                `None`, which will delete those ports.
            skip_port_check: Can be used to skip the internal call to `check_ports`,
                in case it has already been performed elsewhere.
            append: If `True`, `other` is appended instead of being referenced.
                Note that this does not flatten  `other`, so its refs will still
                be refs (now inside `self`).

        Returns:
            self

        Raises:
            `PortError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `other.ports`.
            `PortError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
        """
        if self._dead:
            logger.error('Skipping place() since device is dead')
            return self

        if isinstance(other, str):
            other = self.library.abstract(other)
        if append and isinstance(other, Abstract):
            other = self.library[other.name]

        self.pattern.place(
            other=other,
            offset=offset,
            rotation=rotation,
            pivot=pivot,
            mirrored=mirrored,
            port_map=port_map,
            skip_port_check=skip_port_check,
            append=append,
            )
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
        for port in self.ports.values():
            port.rotate_around(pivot, angle)
        return self

    def mirror(self, axis: int = 0) -> Self:
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
        s = f'<Builder {self.pattern} L({len(self.library)})>'
        return s


