"""
Tools are objects which dynamically generate simple single-use devices (e.g. wires or waveguides)
"""
from typing import Sequence, Literal, Callable, Any
from abc import ABCMeta, abstractmethod     # TODO any way to make Tool ok with implementing only one method?
from dataclasses import dataclass

import numpy
from numpy import pi

from ..utils import SupportsBool, rotation_matrix_2d, layer_t
from ..ports import Port
from ..pattern import Pattern
from ..abstract import Abstract
from ..library import ILibrary, Library
from ..error import BuildError
from .builder import Builder


@dataclass(frozen=True, slots=True)
class RenderStep:
    opcode: Literal['L', 'S', 'U', 'P']
    tool: 'Tool' | None
    start_port: Port
    data: Any

    def __post_init__(self) -> None:
        if self.opcode != 'P' and self.tool is None:
            raise BuildError('Got tool=None but the opcode is not "P"')


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
            ) -> Any:
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
            ) -> Any:
        raise NotImplementedError(f'planS() not implemented for {type(self)}')

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> ILibrary:
        assert not batch or batch[0].tool == self
        raise NotImplementedError(f'render() not implemented for {type(self)}')


abstract_tuple_t = tuple[Abstract, str, str]


class BasicTool(Tool, metaclass=ABCMeta):
    straight: tuple[Callable[[float], Pattern], str, str]
    bend: abstract_tuple_t             # Assumed to be clockwise
    transitions: dict[str, abstract_tuple_t]
    default_out_ptype: str

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
        straight_length, _ccw, in_transition, out_transition = self.planL(
            ccw,
            length,
            in_ptype=in_ptype,
            out_ptype=out_ptype,
            )

        gen_straight, sport_in, sport_out = self.straight
        tree = Library()
        bb = Builder(library=tree, name='_path').add_port_pair(names=port_names)
        if in_transition:
            ipat, iport_theirs, _iport_ours = in_transition
            bb.plug(ipat, {port_names[1]: iport_theirs})
        if not numpy.isclose(straight_length, 0):
            straight = tree << {'_straight': gen_straight(straight_length)}
            bb.plug(straight, {port_names[1]: sport_in})
        if ccw is not None:
            bend, bport_in, bport_out = self.bend
            bb.plug(bend, {port_names[1]: bport_in}, mirrored=(False, bool(ccw)))
        if out_transition:
            opat, oport_theirs, oport_ours = out_transition
            bb.plug(opat, {port_names[1]: oport_ours})

        return bb.pattern

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[float, SupportsBool | None, abstract_tuple_t | None, abstract_tuple_t | None]:
        # TODO check all the math for L-shaped bends
        if ccw is not None:
            bend, bport_in, bport_out = self.bend

            angle_in = bend.ports[bport_in].rotation
            angle_out = bend.ports[bport_out].rotation
            assert angle_in is not None
            assert angle_out is not None

            bend_dxy = rotation_matrix_2d(-angle_in) @ (
                bend.ports[bport_out].offset
                - bend.ports[bport_in].offset
                )

            bend_angle = angle_out - angle_in

            if bool(ccw):
                bend_dxy[1] *= -1
                bend_angle *= -1
        else:
            bend_dxy = numpy.zeros(2)
            bend_angle = 0

        in_transition = self.transitions.get('unk' if in_ptype is None else in_ptype, None)
        if in_transition is not None:
            ipat, iport_theirs, iport_ours = in_transition
            irot = ipat.ports[iport_theirs].rotation
            assert irot is not None
            itrans_dxy = rotation_matrix_2d(-irot) @ (
                ipat.ports[iport_ours].offset
                - ipat.ports[iport_theirs].offset
                )
        else:
            itrans_dxy = numpy.zeros(2)

        out_transition = self.transitions.get('unk' if out_ptype is None else out_ptype, None)
        if out_transition is not None:
            opat, oport_theirs, oport_ours = out_transition
            orot = opat.ports[oport_ours].rotation
            assert orot is not None

            otrans_dxy = rotation_matrix_2d(-orot + bend_angle) @ (
                opat.ports[oport_theirs].offset
                - opat.ports[oport_ours].offset
                )
        else:
            otrans_dxy = numpy.zeros(2)

        if out_transition is not None:
            out_ptype_actual = opat.ports[oport_theirs].ptype
        elif ccw is not None:
            out_ptype_actual = bend.ports[bport_out].ptype
        else:
            out_ptype_actual = self.default_out_ptype

        straight_length = length - bend_dxy[0] - itrans_dxy[0] - otrans_dxy[0]
        bend_run = bend_dxy[1] + itrans_dxy[1] + otrans_dxy

        if straight_length < 0:
            raise BuildError(
                f'Asked to draw path with total length {length:,g}, shorter than required bends and transitions:\n'
                f'bend: {bend_dxy[0]:,g}  in_trans: {itrans_dxy[0]:,g}  out_trans: {otrans_dxy[0]:,g}'
                )

        return float(straight_length), ccw, in_transition, out_transition

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: Sequence[str] = ('A', 'B'),
            append: bool = True,
            **kwargs,
            ) -> ILibrary:

        tree = Library()
        bb = Builder(library=tree, name='_path').add_port_pair(names=(port_names[0], port_names[1]))

        gen_straight, sport_in, _sport_out = self.straight
        for step in batch:
            straight_length, ccw, in_transition, out_transition = step.data
            assert step.tool == self

            if step.opcode == 'L':
                if in_transition:
                    ipat, iport_theirs, _iport_ours = in_transition
                    bb.plug(ipat, {port_names[1]: iport_theirs})
                if not numpy.isclose(straight_length, 0):
                    straight_pat = gen_straight(straight_length)
                    if append:
                        bb.plug(straight_pat, {port_names[1]: sport_in}, append=True)
                    else:
                        straight = tree << {'_straight': straight_pat}
                        bb.plug(straight, {port_names[1]: sport_in}, append=True)
                if ccw is not None:
                    bend, bport_in, bport_out = self.bend
                    bb.plug(bend, {port_names[1]: bport_in}, mirrored=(False, bool(ccw)))
                if out_transition:
                    opat, oport_theirs, oport_ours = out_transition
                    bb.plug(opat, {port_names[1]: oport_ours})
        return tree



class PathTool(Tool, metaclass=ABCMeta):
    straight: tuple[Callable[[float], Pattern], str, str]
    bend: abstract_tuple_t             # Assumed to be clockwise
    transitions: dict[str, abstract_tuple_t]
    ptype: str
    width: float
    layer: layer_t

    def __init__(self, layer: layer_t, width: float, ptype: str = 'unk') -> None:
        Tool.__init__(self)
        self.layer = layer
        self.width = width
        self.ptype: str

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
        dxy = self.planL(
            ccw,
            length,
            in_ptype=in_ptype,
            out_ptype=out_ptype,
            )

        pat = Pattern()
        pat.path(layer=self.layer, width=self.width, vertices=[(0, 0), (length, 0)])

        if ccw is None:
            out_rot = pi
        elif bool(ccw):
            out_rot = -pi / 2
        else:
            out_rot = pi / 2

        pat.ports = {
            port_names[0]: Port((0, 0), rotation=0, ptype=self.ptype),
            port_names[1]: Port(dxy, rotation=out_rot, ptype=self.ptype),
            }

        return pat

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[float, float]:
        # TODO check all the math for L-shaped bends

        if out_ptype and out_ptype != self.ptype:
            raise BuildError(f'Requested {out_ptype=} does not match path ptype {self.ptype}')

        if ccw is not None:
            bend_dxy = numpy.array([1, -1]) * self.width / 2
            bend_angle = pi / 2

            if bool(ccw):
                bend_dxy[1] *= -1
                bend_angle *= -1
        else:
            bend_dxy = numpy.zeros(2)
            bend_angle = 0

        straight_length = length - bend_dxy[0]
        bend_run = bend_dxy[1]

        if straight_length < 0:
            raise BuildError(
                f'Asked to draw path with total length {length:,g}, shorter than required bend: {bend_dxy[0]:,g}'
                )

        return length, bend_run

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> ILibrary:

        path_vertices = [batch[0].start_port.offset]
        for step in batch:
            assert step.tool == self

            port_rot = step.start_port.rotation
            assert port_rot is not None

            if step.opcode == 'L':
                length, bend_run = step.data
                dxy = rotation_matrix_2d(port_rot + pi) @ (length, bend_run)
                path_vertices.append(step.start_port.offset + dxy)
            else:
                raise BuildError(f'Unrecognized opcode "{step.opcode}"')

        tree, pat = Library.mktree('_path')
        pat.path(layer=self.layer, width=self.width, vertices=path_vertices)

        return tree
