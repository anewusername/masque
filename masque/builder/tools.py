"""
Tools are objects which dynamically generate simple single-use devices (e.g. wires or waveguides)

# TODO document all tools
"""
from typing import Sequence, Literal, Callable, Any
from abc import ABCMeta  # , abstractmethod     # TODO any way to make Tool ok with implementing only one method?
from dataclasses import dataclass

import numpy
from numpy.typing import NDArray
from numpy import pi

from ..utils import SupportsBool, rotation_matrix_2d, layer_t
from ..ports import Port
from ..pattern import Pattern
from ..abstract import Abstract
from ..library import ILibrary, Library, SINGLE_USE_PREFIX
from ..error import BuildError


@dataclass(frozen=True, slots=True)
class RenderStep:
    """
    Representation of a single saved operation, used by `RenderPather` and passed
    to `Tool.render()` when `RenderPather.render()` is called.
    """
    opcode: Literal['L', 'S', 'U', 'P']
    """ What operation is being performed.
        L: planL   (straight, optionally with a single bend)
        S: planS   (s-bend)
        U: planU   (u-bend)
        P: plug
    """

    tool: 'Tool | None'
    """ The current tool. May be `None` if `opcode='P'` """

    start_port: Port
    end_port: Port

    data: Any
    """ Arbitrary tool-specific data"""

    def __post_init__(self) -> None:
        if self.opcode != 'P' and self.tool is None:
            raise BuildError('Got tool=None but the opcode is not "P"')


class Tool:
    """
    Interface for path (e.g. wire or waveguide) generation.

    Note that subclasses may implement only a subset of the methods and leave others
    unimplemented (e.g. in cases where they don't make sense or the required components
    are impractical or unavailable).
    """
    def path(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Library:
        """
        Create a wire or waveguide that travels exactly `length` distance along the axis
        of its input port.

        Used by `Pather`.

        The output port must be exactly `length` away along the input port's axis, but
        may be placed an additional (unspecified) distance away along the perpendicular
        direction. The output port should be rotated (or not) based on the value of
        `ccw`.

        The input and output ports should be compatible with `in_ptype` and
        `out_ptype`, respectively. They should also be named `port_names[0]` and
        `port_names[1]`, respectively.

        Args:
            ccw: If `None`, the output should be along the same axis as the input.
                Otherwise, cast to bool and turn counterclockwise if True
                and clockwise otherwise.
            length: The total distance from input to output, along the input's axis only.
                (There may be a tool-dependent offset along the other axis.)
            in_ptype: The `ptype` of the port into which this wire's input will be `plug`ged.
            out_ptype: The `ptype` of the port into which this wire's output will be `plug`ged.
            port_names: The output pattern will have its input port named `port_names[0]` and
                its output named `port_names[1]`.
            kwargs: Custom tool-specific parameters.

        Returns:
            A pattern tree containing the requested L-shaped (or straight) wire or waveguide

        Raises:
            BuildError if an impossible or unsupported geometry is requested.
        """
        raise NotImplementedError(f'path() not implemented for {type(self)}')

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[Port, Any]:
        """
        Plan a wire or waveguide that travels exactly `length` distance along the axis
        of its input port.

        Used by `RenderPather`.

        The output port must be exactly `length` away along the input port's axis, but
        may be placed an additional (unspecified) distance away along the perpendicular
        direction. The output port should be rotated (or not) based on the value of
        `ccw`.

        The input and output ports should be compatible with `in_ptype` and
        `out_ptype`, respectively.

        Args:
            ccw: If `None`, the output should be along the same axis as the input.
                Otherwise, cast to bool and turn counterclockwise if True
                and clockwise otherwise.
            length: The total distance from input to output, along the input's axis only.
                (There may be a tool-dependent offset along the other axis.)
            in_ptype: The `ptype` of the port into which this wire's input will be `plug`ged.
            out_ptype: The `ptype` of the port into which this wire's output will be `plug`ged.
            kwargs: Custom tool-specific parameters.

        Returns:
            The calculated output `Port` for the wire.
            Any tool-specifc data, to be stored in `RenderStep.data`, for use during rendering.

        Raises:
            BuildError if an impossible or unsupported geometry is requested.
        """
        raise NotImplementedError(f'planL() not implemented for {type(self)}')

    def planS(
            self,
            length: float,
            jog: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[Port, Any]:
        """
        Plan a wire or waveguide that travels exactly `length` distance along the axis
        of its input port and `jog` distance along the perpendicular axis (i.e. an S-bend).

        Used by `RenderPather`.

        The output port must have an orientation rotated by pi from the input port.

        The input and output ports should be compatible with `in_ptype` and
        `out_ptype`, respectively.

        Args:
            length: The total distance from input to output, along the input's axis only.
            jog: The total offset from the input to output, along the perpendicular axis.
                A positive number implies a rightwards shift (i.e. clockwise bend followed
                by a counterclockwise bend)
            in_ptype: The `ptype` of the port into which this wire's input will be `plug`ged.
            out_ptype: The `ptype` of the port into which this wire's output will be `plug`ged.
            kwargs: Custom tool-specific parameters.

        Returns:
            The calculated output `Port` for the wire.
            Any tool-specifc data, to be stored in `RenderStep.data`, for use during rendering.

        Raises:
            BuildError if an impossible or unsupported geometry is requested.
        """
        raise NotImplementedError(f'planS() not implemented for {type(self)}')

    def planU(
            self,
            jog: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[Port, Any]:
        """
        # NOTE: TODO: U-bend is WIP; this interface may change in the future.

        Plan a wire or waveguide that travels exactly `jog` distance along the axis
        perpendicular to its input port (i.e. a U-bend).

        Used by `RenderPather`.

        The output port must have an orientation identical to the input port.

        The input and output ports should be compatible with `in_ptype` and
        `out_ptype`, respectively.

        Args:
            jog: The total offset from the input to output, along the perpendicular axis.
                A positive number implies a rightwards shift (i.e. clockwise bend followed
                by a counterclockwise bend)
            in_ptype: The `ptype` of the port into which this wire's input will be `plug`ged.
            out_ptype: The `ptype` of the port into which this wire's output will be `plug`ged.
            kwargs: Custom tool-specific parameters.

        Returns:
            The calculated output `Port` for the wire.
            Any tool-specifc data, to be stored in `RenderStep.data`, for use during rendering.

        Raises:
            BuildError if an impossible or unsupported geometry is requested.
        """
        raise NotImplementedError(f'planU() not implemented for {type(self)}')

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: Sequence[str] = ('A', 'B'),
            **kwargs,
            ) -> ILibrary:
        """
        Render the provided `batch` of `RenderStep`s into geometry, returning a tree
        (a Library with a single topcell).

        Args:
            batch: A sequence of `RenderStep` objects containing the ports and data
                provided by this tool's `planL`/`planS`/`planU` functions.
            port_names: The topcell's input and output ports should be named
                `port_names[0]` and `port_names[1]` respectively.
            kwargs: Custom tool-specific parameters.
        """
        assert not batch or batch[0].tool == self
        raise NotImplementedError(f'render() not implemented for {type(self)}')


abstract_tuple_t = tuple[Abstract, str, str]


@dataclass
class BasicTool(Tool, metaclass=ABCMeta):
    """
      A simple tool which relies on a single pre-rendered `bend` pattern, a function
    for generating straight paths, and a table of pre-rendered `transitions` for converting
    from non-native ptypes.
    """
    straight: tuple[Callable[[float], Pattern], str, str]
    """ `create_straight(length: float), in_port_name, out_port_name` """

    bend: abstract_tuple_t             # Assumed to be clockwise
    """ `clockwise_bend_abstract, in_port_name, out_port_name` """

    transitions: dict[str, abstract_tuple_t]
    """ `{ptype: (transition_abstract`, ptype_port_name, other_port_name), ...}` """

    default_out_ptype: str
    """ Default value for out_ptype """

    @dataclass(frozen=True, slots=True)
    class LData:
        """ Data for planL """
        straight_length: float
        ccw: SupportsBool | None
        in_transition: abstract_tuple_t | None
        out_transition: abstract_tuple_t | None

    def path(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Library:
        _out_port, data = self.planL(
            ccw,
            length,
            in_ptype=in_ptype,
            out_ptype=out_ptype,
            )

        gen_straight, sport_in, sport_out = self.straight
        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
        pat.add_port_pair(names=port_names)
        if data.in_transition:
            ipat, iport_theirs, _iport_ours = data.in_transition
            pat.plug(ipat, {port_names[1]: iport_theirs})
        if not numpy.isclose(data.straight_length, 0):
            straight = tree <= {SINGLE_USE_PREFIX + 'straight': gen_straight(data.straight_length)}
            pat.plug(straight, {port_names[1]: sport_in})
        if data.ccw is not None:
            bend, bport_in, bport_out = self.bend
            pat.plug(bend, {port_names[1]: bport_in}, mirrored=bool(ccw))
        if data.out_transition:
            opat, oport_theirs, oport_ours = data.out_transition
            pat.plug(opat, {port_names[1]: oport_ours})

        return tree

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[Port, LData]:
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
        bend_run = bend_dxy[1] + itrans_dxy[1] + otrans_dxy[1]

        if straight_length < 0:
            raise BuildError(
                f'Asked to draw path with total length {length:,g}, shorter than required bends and transitions:\n'
                f'bend: {bend_dxy[0]:,g}  in_trans: {itrans_dxy[0]:,g}  out_trans: {otrans_dxy[0]:,g}'
                )

        data = self.LData(straight_length, ccw, in_transition, out_transition)
        out_port = Port((length, bend_run), rotation=bend_angle, ptype=out_ptype_actual)
        return out_port, data

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: Sequence[str] = ('A', 'B'),
            append: bool = True,
            **kwargs,
            ) -> ILibrary:

        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
        pat.add_port_pair(names=(port_names[0], port_names[1]))

        gen_straight, sport_in, _sport_out = self.straight
        for step in batch:
            straight_length, ccw, in_transition, out_transition = step.data
            assert step.tool == self

            if step.opcode == 'L':
                if in_transition:
                    ipat, iport_theirs, _iport_ours = in_transition
                    pat.plug(ipat, {port_names[1]: iport_theirs})
                if not numpy.isclose(straight_length, 0):
                    straight_pat = gen_straight(straight_length)
                    if append:
                        pat.plug(straight_pat, {port_names[1]: sport_in}, append=True)
                    else:
                        straight = tree <= {SINGLE_USE_PREFIX + 'straight': straight_pat}
                        pat.plug(straight, {port_names[1]: sport_in}, append=True)
                if ccw is not None:
                    bend, bport_in, bport_out = self.bend
                    pat.plug(bend, {port_names[1]: bport_in}, mirrored=bool(ccw))
                if out_transition:
                    opat, oport_theirs, oport_ours = out_transition
                    pat.plug(opat, {port_names[1]: oport_ours})
        return tree


@dataclass
class PathTool(Tool, metaclass=ABCMeta):
    """
    A tool which draws `Path` geometry elements.

    If `planL` / `render` are used, the `Path` elements can cover >2 vertices;
    with `path` only individual rectangles will be drawn.
    """
    layer: layer_t
    """ Layer to draw on """

    width: float
    """ `Path` width """

    ptype: str = 'unk'
    """ ptype for any ports in patterns generated by this tool """

    #@dataclass(frozen=True, slots=True)
    #class LData:
    #    dxy: NDArray[numpy.float64]

    #def __init__(self, layer: layer_t, width: float, ptype: str = 'unk') -> None:
    #    Tool.__init__(self)
    #    self.layer = layer
    #    self.width = width
    #    self.ptype: str

    def path(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Library:
        out_port, dxy = self.planL(
            ccw,
            length,
            in_ptype=in_ptype,
            out_ptype=out_ptype,
            )

        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
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

        return tree

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[Port, NDArray[numpy.float64]]:
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
            bend_angle = pi

        straight_length = length - bend_dxy[0]
        bend_run = bend_dxy[1]

        if straight_length < 0:
            raise BuildError(
                f'Asked to draw path with total length {length:,g}, shorter than required bend: {bend_dxy[0]:,g}'
                )
        data = numpy.array((length, bend_run))
        out_port = Port(data, rotation=bend_angle, ptype=self.ptype)
        return out_port, data

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
                dxy = rotation_matrix_2d(port_rot + pi) @ (length, 0)
                #path_vertices.append(step.start_port.offset)
                path_vertices.append(step.start_port.offset + dxy)
            else:
                raise BuildError(f'Unrecognized opcode "{step.opcode}"')

        if (path_vertices[-1] != batch[-1].end_port.offset).any():
            # If the path ends in a bend, we need to add the final vertex
            path_vertices.append(batch[-1].end_port.offset)

        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
        pat.path(layer=self.layer, width=self.width, vertices=path_vertices)
        pat.ports = {
            port_names[0]: batch[0].start_port.copy().rotate(pi),
            port_names[1]: batch[-1].end_port.copy().rotate(pi),
            }
        return tree
