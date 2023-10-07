from typing import Self
import copy
import logging

import numpy
from numpy.typing import ArrayLike

from .ref import Ref
from .ports import PortList, Port
from .utils import rotation_matrix_2d

#if TYPE_CHECKING:
#    from .builder import Builder, Tool
#    from .library import ILibrary


logger = logging.getLogger(__name__)


class Abstract(PortList):
    """
    An `Abstract` is a container for a name and associated ports.

    When snapping a sub-component to an existing pattern, only the name (not contained
    in a `Pattern` object) and port info is needed, and not the geometry itself.
    """
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

    # TODO do we want to store a Ref instead of just a name? then we can translate/rotate/mirror...

    def __repr__(self) -> str:
        s = f'<Abstract {self.name} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s

    def translate_ports(self, offset: ArrayLike) -> Self:
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

    def scale_by(self, c: float) -> Self:
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

    def rotate_around(self, pivot: ArrayLike, rotation: float) -> Self:
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

    def rotate_port_offsets(self, rotation: float) -> Self:
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

    def rotate_ports(self, rotation: float) -> Self:
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

    def mirror_port_offsets(self, across_axis: int = 0) -> Self:
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

    def mirror_ports(self, across_axis: int = 0) -> Self:
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

    def mirror(self, across_axis: int = 0) -> Self:
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

    def apply_ref_transform(self, ref: Ref) -> Self:
        """
        Apply the transform from a `Ref` to the ports of this `Abstract`.
        This changes the port locations to where they would be in the Ref's parent pattern.

        Args:
            ref: The ref whose transform should be applied.

        Returns:
            self
        """
        if ref.mirrored:
            self.mirror()
        self.rotate_ports(ref.rotation)
        self.rotate_port_offsets(ref.rotation)
        self.translate_ports(ref.offset)
        return self

    def undo_ref_transform(self, ref: Ref) -> Self:
        """
        Apply the inverse transform from a `Ref` to the ports of this `Abstract`.
        This changes the port locations to where they would be in the Ref's target (from the parent).

        Args:
            ref: The ref whose (inverse) transform should be applied.

        Returns:
            self

        # TODO test undo_ref_transform
        """
        self.translate_ports(-ref.offset)
        self.rotate_port_offsets(-ref.rotation)
        self.rotate_ports(-ref.rotation)
        if ref.mirrored:
            self.mirror(0)
        return self
