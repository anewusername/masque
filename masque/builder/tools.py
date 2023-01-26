"""
Tools are objects which dynamically generate simple single-use devices (e.g. wires or waveguides)
"""
from typing import TYPE_CHECKING, Optional, Sequence

from ..utils import SupportsBool

if TYPE_CHECKING:
    from ..pattern import Pattern


class Tool:
    def path(
            self,
            ccw: Optional[SupportsBool],
            length: float,
            *,
            in_ptype: Optional[str] = None,
            out_ptype: Optional[str] = None,
            port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> 'Pattern':
        raise NotImplementedError(f'path() not implemented for {type(self)}')

