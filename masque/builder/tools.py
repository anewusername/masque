"""
Tools are objects which dynamically generate simple single-use devices (e.g. wires or waveguides)
"""
from typing import TYPE_CHECKING, Sequence, Literal, Tuple

from ..utils import SupportsBool
from ..ports import Port
from ..pattern import Pattern
from ..library import ILibrary


render_step_t = (
    tuple[Literal['L', 'S', 'U'], Port, float, float, str, 'Tool']
    | tuple[Literal['P'], None, float, float, str, None]
    )

class Tool:
    def path(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> Pattern:
        raise NotImplementedError(f'path() not implemented for {type(self)}')

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> Tuple[float, str]:
        raise NotImplementedError(f'planL() not implemented for {type(self)}')

    def planS(
            self,
            ccw: SupportsBool | None,
            length: float,
            jog: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> str:       # out_ptype only?
        raise NotImplementedError(f'planS() not implemented for {type(self)}')

    def render(
            self,
            batch: Sequence[render_step_t],
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> ILibrary:
        assert batch[0][-1] == self
        raise NotImplementedError(f'render() not implemented for {type(self)}')

