"""
Tools are objects which dynamically generate simple single-use devices (e.g. wires or waveguides)
"""
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from .devices import Device


class Tool:
    def path(
            self,
            ccw: Optional[bool],
            length: float,
            *,
            in_ptype: Optional[str] = None,
            out_ptype: Optional[str] = None,
            port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> 'Device':
        raise NotImplementedError(f'path() not implemented for {type(self)}')

