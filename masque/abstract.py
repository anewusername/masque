from typing import TypeVar
import copy
import logging

import numpy
from numpy.typing import ArrayLike

from .ref import Ref
from .ports import PortList, Port
from .utils import rotation_matrix_2d, normalize_mirror

#if TYPE_CHECKING:
#    from .builder import Builder, Tool
#    from .library import MutableLibrary


logger = logging.getLogger(__name__)


AA  = TypeVar('AA',  bound='Abstract')


class Abstract(PortList):
    __slots__ = ('name', '_ports')

    name: str
    """ Name of the pattern this device references """

    _ports: dict[str, Port]
    """ Uniquely-named ports which can be used to instances together"""

    @property
    def ports(self) -> dict[str, Port]:
        return self._ports

    @ports.setter
    def ports(self, value: dict[str, Port]) -> None:
        self._ports = value

    def __init__(
            self,
            name: str,
            ports: dict[str, Port],
            ) -> None:
        self.name = name
        self.ports = copy.deepcopy(ports)

#    def build(
#            self,
#            library: 'MutableLibrary',
#            tools: None | 'Tool' | MutableMapping[str | None, 'Tool'] = None,
#            ) -> 'Builder':
#        """
#        Begin building a new device around an instance of the current device
#          (rather than modifying the current device).
#
#        Returns:
#            The new `Builder` object.
#        """
#        pat = Pattern(ports=self.ports)
#        pat.ref(self.name)
#        new = Builder(library=library, pattern=pat, tools=tools)   # TODO should Abstract have tools?
#        return new

    # TODO do we want to store a Ref instead of just a name? then we can translate/rotate/mirror...

    def __repr__(self) -> str:
        s = f'<Abstract {self.name} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s

    def translate_ports(self: AA, offset: ArrayLike) -> AA:
        """
        Translates all ports by the given offset.

        Args:
            offset: (x, y) to translate by

        Returns:
            self
        """
        for port in self.ports.values():
            port.translate(offset)
        return self

    def scale_by(self: AA, c: float) -> AA:
        """
        Scale this Abstract by the given value
         (all port offsets are scaled)

        Args:
            c: factor to scale by

        Returns:
            self
        """
        for port in self.ports.values():
            port.offset *= c
        return self

    def rotate_around(self: AA, pivot: ArrayLike, rotation: float) -> AA:
        """
        Rotate the Abstract around the a location.

        Args:
            pivot: (x, y) location to rotate around
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        pivot = numpy.array(pivot)
        self.translate_ports(-pivot)
        self.rotate_ports(rotation)
        self.rotate_port_offsets(rotation)
        self.translate_ports(+pivot)
        return self

    def rotate_port_offsets(self: AA, rotation: float) -> AA:
        """
        Rotate the offsets of all ports around (0, 0)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for port in self.ports.values():
            port.offset = rotation_matrix_2d(rotation) @ port.offset
        return self

    def rotate_ports(self: AA, rotation: float) -> AA:
        """
        Rotate each port around its offset (i.e. in place)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for port in self.ports.values():
            port.rotate(rotation)
        return self

    def mirror_port_offsets(self: AA, across_axis: int) -> AA:
        """
        Mirror the offsets of all shapes, labels, and refs across an axis

        Args:
            across_axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for port in self.ports.values():
            port.offset[across_axis - 1] *= -1
        return self

    def mirror_ports(self: AA, across_axis: int) -> AA:
        """
        Mirror each port's rotation across an axis, relative to its
          offset

        Args:
            across_axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for port in self.ports.values():
            port.mirror(across_axis)
        return self

    def mirror(self: AA, across_axis: int) -> AA:
        """
        Mirror the Pattern across an axis

        Args:
            axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        self.mirror_ports(across_axis)
        self.mirror_port_offsets(across_axis)
        return self

    def apply_ref_transform(self: AA, ref: Ref) -> AA:
        """
        Apply the transform from a `Ref` to the ports of this `Abstract`.
        This changes the port locations to where they would be in the Ref's parent pattern.

        Args:
            ref: The ref whose transform should be applied.

        Returns:
            self
        """
        mirrored_across_x, angle = normalize_mirror(ref.mirrored)
        if mirrored_across_x:
            self.mirror(across_axis=0)
        self.rotate_ports(angle + ref.rotation)
        self.rotate_port_offsets(angle + ref.rotation)
        self.translate_ports(ref.offset)
        return self

    def undo_ref_transform(self: AA, ref: Ref) -> AA:
        """
        Apply the inverse transform from a `Ref` to the ports of this `Abstract`.
        This changes the port locations to where they would be in the Ref's target (from the parent).

        Args:
            ref: The ref whose (inverse) transform should be applied.

        Returns:
            self

        # TODO test undo_ref_transform
        """
        mirrored_across_x, angle = normalize_mirror(ref.mirrored)
        self.translate_ports(-ref.offset)
        self.rotate_port_offsets(-angle - ref.rotation)
        self.rotate_ports(-angle - ref.rotation)
        if mirrored_across_x:
            self.mirror(across_axis=0)
        return self
