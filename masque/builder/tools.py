"""
Tools are objects which dynamically generate simple single-use devices (e.g. wires or waveguides)
"""
from typing import Sequence, Literal, Callable
from abc import ABCMeta, abstractmethod

import numpy

from ..utils import SupportsBool, rotation_matrix_2d
from ..ports import Port
from ..pattern import Pattern
from ..abstract import Abstract
from ..library import ILibrary, Library
from ..error import BuildError
from .builder import Builder


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
            port_names: tuple[str, str] = ('A', 'B'),
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
            ) -> tuple[float, str]:
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


class BasicTool(Tool, metaclass=ABCMeta):
    straight: tuple[Callable[[float], Pattern], str, str]
    bend: tuple[Abstract, str, str]             # Assumed to be clockwise
    transitions: dict[str, tuple[Abstract, str, str]]

    def path(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Pattern:

        # TODO check all the math for L-shaped bends
        straight_length = length
        bend_run = 0
        if ccw is not None:
            bend, bport_in, bport_out = self.bend
            brot = bend.ports[bport_in].rotation
            assert brot is not None
            bend_dxy = numpy.abs(
                rotation_matrix_2d(-brot) @ (
                    bend.ports[bport_out].offset
                    - bend.ports[bport_in].offset
                    )
                )

            straight_length -= bend_dxy[0]
            bend_run += bend_dxy[1]
        else:
            bend_dxy = numpy.zeros(2)

        in_transition = self.transitions.get('unk' if in_ptype is None else in_ptype, None)
        if in_transition is not None:
            ipat, iport_theirs, iport_ours = in_transition
            irot = ipat.ports[iport_theirs].rotation
            assert irot is not None
            itrans_dxy = rotation_matrix_2d(-irot) @ (
                ipat.ports[iport_ours].offset
                - ipat.ports[iport_theirs].offset
                )

            straight_length -= itrans_dxy[0]
            bend_run += itrans_dxy[1]
        else:
            itrans_dxy = numpy.zeros(2)

        out_transition = self.transitions.get('unk' if out_ptype is None else out_ptype, None)
        if out_transition is not None:
            opat, oport_theirs, oport_ours = out_transition
            orot = opat.ports[oport_ours].rotation
            assert orot is not None
            otrans_dxy = rotation_matrix_2d(-orot) @ (
                opat.ports[oport_theirs].offset
                - opat.ports[oport_ours].offset
                )
            if ccw:
                otrans_dxy[0] *= -1

            straight_length -= otrans_dxy[1]
            bend_run += otrans_dxy[0]
        else:
            otrans_dxy = numpy.zeros(2)

        if straight_length < 0:
            raise BuildError(f'Asked to draw path with total length {length:g}, shorter than required bends and tapers:\n'
                             f'bend: {bend_dxy[0]:g}  in_taper: {abs(itrans_dxy[0])}  out_taper: {otrans_dxy[1]}')

        gen_straight, sport_in, sport_out = self.straight
        tree = Library()
        bb = Builder(library=tree, name='_path').add_port_pair(names=port_names)
        if in_transition:
            bb.plug(ipat, {port_names[1]: iport_theirs})
        if not numpy.isclose(straight_length, 0):
            straight = tree << {'_straight': gen_straight(straight_length)}
            bb.plug(straight, {port_names[1]: sport_in})
        if ccw is not None:
            bb.plug(bend, {port_names[1]: bport_in}, mirrored=(False, bool(ccw)))
        if out_transition:
            bb.plug(opat, {port_names[1]: oport_ours})

        return bb.pattern
