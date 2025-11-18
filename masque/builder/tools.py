"""
Tools are objects which dynamically generate simple single-use devices (e.g. wires or waveguides)

# TODO document all tools
"""
from typing import Literal, Any, Self
from collections.abc import Sequence, Callable
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

        Used by `Pather` and `RenderPather`.

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

    def pathS(
            self,
            length: float,
            jog: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Library:
        """
        Create a wire or waveguide that travels exactly `length` distance along the axis
        of its input port, and `jog` distance on the perpendicular axis.
        `jog` is positive when moving left of the direction of travel (from input to ouput port).

        Used by `Pather` and `RenderPather`.

        The output port should be rotated to face the input port (i.e. plugging the device
        into a port will move that port but keep its orientation).

        The input and output ports should be compatible with `in_ptype` and
        `out_ptype`, respectively. They should also be named `port_names[0]` and
        `port_names[1]`, respectively.

        Args:
            length: The total distance from input to output, along the input's axis only.
            jog: The total distance from input to output, along the second axis. Positive indicates
                a leftward shift when moving from input to output port.
            in_ptype: The `ptype` of the port into which this wire's input will be `plug`ged.
            out_ptype: The `ptype` of the port into which this wire's output will be `plug`ged.
            port_names: The output pattern will have its input port named `port_names[0]` and
                its output named `port_names[1]`.
            kwargs: Custom tool-specific parameters.

        Returns:
            A pattern tree containing the requested S-shaped (or straight) wire or waveguide

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
            The calculated output `Port` for the wire, assuming an input port at (0, 0) with rotation 0.
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
            The calculated output `Port` for the wire, assuming an input port at (0, 0) with rotation 0.
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
                A positive number implies a leftwards shift (i.e. counterclockwise bend
                followed by a clockwise bend)
            in_ptype: The `ptype` of the port into which this wire's input will be `plug`ged.
            out_ptype: The `ptype` of the port into which this wire's output will be `plug`ged.
            kwargs: Custom tool-specific parameters.

        Returns:
            The calculated output `Port` for the wire, assuming an input port at (0, 0) with rotation 0.
            Any tool-specifc data, to be stored in `RenderStep.data`, for use during rendering.

        Raises:
            BuildError if an impossible or unsupported geometry is requested.
        """
        raise NotImplementedError(f'planU() not implemented for {type(self)}')

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: tuple[str, str] = ('A', 'B'),     # noqa: ARG002 (unused)
            **kwargs,                                   # noqa: ARG002 (unused)
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
class SimpleTool(Tool, metaclass=ABCMeta):
    """
      A simple tool which relies on a single pre-rendered `bend` pattern, a function
    for generating straight paths, and a table of pre-rendered `transitions` for converting
    from non-native ptypes.
    """
    straight: tuple[Callable[[float], Pattern] | Callable[[float], Library], str, str]
    """ `create_straight(length: float), in_port_name, out_port_name` """

    bend: abstract_tuple_t             # Assumed to be clockwise
    """ `clockwise_bend_abstract, in_port_name, out_port_name` """

    default_out_ptype: str
    """ Default value for out_ptype """

    @dataclass(frozen=True, slots=True)
    class LData:
        """ Data for planL """
        straight_length: float
        straight_kwargs: dict[str, Any]
        ccw: SupportsBool | None

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,   # noqa: ARG002 (unused)
            out_ptype: str | None = None,  # noqa: ARG002 (unused)
            **kwargs,               # noqa: ARG002 (unused)
            ) -> tuple[Port, LData]:
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
            bend_angle = pi

        if ccw is not None:
            out_ptype_actual = bend.ports[bport_out].ptype
        else:
            out_ptype_actual = self.default_out_ptype

        straight_length = length - bend_dxy[0]
        bend_run = bend_dxy[1]

        if straight_length < 0:
            raise BuildError(
                f'Asked to draw L-path with total length {length:,g}, shorter than required bends ({bend_dxy[0]:,})'
                )

        data = self.LData(straight_length, kwargs, ccw)
        out_port = Port((length, bend_run), rotation=bend_angle, ptype=out_ptype_actual)
        return out_port, data

    def _renderL(
            self,
            data: LData,
            tree: ILibrary,
            port_names: tuple[str, str],
            straight_kwargs: dict[str, Any],
            ) -> ILibrary:
        """
        Render an L step into a preexisting tree
        """
        pat = tree.top_pattern()
        gen_straight, sport_in, _sport_out = self.straight
        if not numpy.isclose(data.straight_length, 0):
            straight_pat_or_tree = gen_straight(data.straight_length, **(straight_kwargs | data.straight_kwargs))
            pmap = {port_names[1]: sport_in}
            if isinstance(straight_pat_or_tree, Pattern):
                straight_pat = straight_pat_or_tree
                pat.plug(straight_pat, pmap, append=True)
            else:
                straight_tree = straight_pat_or_tree
                top = straight_tree.top()
                straight_tree.flatten(top)
                pat.plug(straight_tree[top], pmap, append=True)
        if data.ccw is not None:
            bend, bport_in, bport_out = self.bend
            pat.plug(bend, {port_names[1]: bport_in}, mirrored=bool(data.ccw))
        return tree

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
            in_ptype = in_ptype,
            out_ptype = out_ptype,
            )

        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
        pat.add_port_pair(names=port_names, ptype='unk' if in_ptype is None else in_ptype)
        self._renderL(data=data, tree=tree, port_names=port_names, straight_kwargs=kwargs)
        return tree

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> ILibrary:

        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
        pat.add_port_pair(names=(port_names[0], port_names[1]))

        for step in batch:
            assert step.tool == self
            if step.opcode == 'L':
                self._renderL(data=step.data, tree=tree, port_names=port_names, straight_kwargs=kwargs)
        return tree


@dataclass
class AutoTool(Tool, metaclass=ABCMeta):
    """
      A simple tool which relies on a single pre-rendered `bend` pattern, a function
    for generating straight paths, and a table of pre-rendered `transitions` for converting
    from non-native ptypes.
    """

    @dataclass(frozen=True, slots=True)
    class Straight:
        """ Description of a straight-path generator """
        ptype: str
        fn: Callable[[float], Pattern] | Callable[[float], Library]
        in_port_name: str
        out_port_name: str
        length_range: tuple[float, float] = (0, numpy.inf)

    @dataclass(frozen=True, slots=True)
    class SBend:
        """ Description of an s-bend generator """
        ptype: str

        fn: Callable[[float], Pattern] | Callable[[float], Library]
        """
        Generator function. `jog` (only argument) is assumed to be left (ccw) relative to travel
        and may be negative for a jog in the opposite direction. Won't be called if jog=0.
        """

        in_port_name: str
        out_port_name: str
        jog_range: tuple[float, float] = (0, numpy.inf)

    @dataclass(frozen=True, slots=True)
    class Bend:
        """ Description of a pre-rendered bend """
        abstract: Abstract
        in_port_name: str
        out_port_name: str
        clockwise: bool = True

        @property
        def in_port(self) -> Port:
            return self.abstract.ports[self.in_port_name]

        @property
        def out_port(self) -> Port:
            return self.abstract.ports[self.out_port_name]

    @dataclass(frozen=True, slots=True)
    class Transition:
        """ Description of a pre-rendered transition """
        abstract: Abstract
        their_port_name: str
        our_port_name: str

        @property
        def our_port(self) -> Port:
            return self.abstract.ports[self.our_port_name]

        @property
        def their_port(self) -> Port:
            return self.abstract.ports[self.their_port_name]

        def reversed(self) -> Self:
            return type(self)(self.abstract, self.our_port_name, self.their_port_name)

    @dataclass(frozen=True, slots=True)
    class LData:
        """ Data for planL """
        straight_length: float
        straight: 'AutoTool.Straight'
        straight_kwargs: dict[str, Any]
        ccw: SupportsBool | None
        bend: 'AutoTool.Bend | None'
        in_transition: 'AutoTool.Transition | None'
        b_transition: 'AutoTool.Transition | None'
        out_transition: 'AutoTool.Transition | None'

    @dataclass(frozen=True, slots=True)
    class SData:
        """ Data for planS """
        straight_length: float
        straight: 'AutoTool.Straight'
        gen_kwargs: dict[str, Any]
        jog_remaining: float
        sbend: 'AutoTool.SBend'
        in_transition: 'AutoTool.Transition | None'
        b_transition: 'AutoTool.Transition | None'
        out_transition: 'AutoTool.Transition | None'

    straights: list[Straight]
    """ List of straight-generators to choose from, in order of priority """

    bends: list[Bend]
    """ List of bends to choose from, in order of priority """

    sbends: list[SBend]
    """ List of S-bend generators to choose from, in order of priority """

    transitions: dict[tuple[str, str], Transition]
    """ `{(external_ptype, internal_ptype): Transition, ...}` """

    default_out_ptype: str
    """ Default value for out_ptype """

    def add_complementary_transitions(self) -> Self:
        for iioo in list(self.transitions.keys()):
            ooii = (iioo[1], iioo[0])
            self.transitions.setdefault(ooii, self.transitions[iioo].reversed())
        return self

    @staticmethod
    def _bend2dxy(bend: Bend, ccw: SupportsBool | None) -> tuple[NDArray[numpy.float64], float]:
        if ccw is None:
            return numpy.zeros(2), pi
        bend_dxy, bend_angle = bend.in_port.measure_travel(bend.out_port)
        assert bend_angle is not None
        if bool(ccw):
            bend_dxy[1] *= -1
            bend_angle *= -1
        return bend_dxy, bend_angle

    @staticmethod
    def _sbend2dxy(sbend: SBend, jog: float) -> NDArray[numpy.float64]:
        if numpy.isclose(jog, 0):
            return numpy.zeros(2)

        sbend_pat_or_tree = sbend.fn(abs(jog))
        sbpat = sbend_pat_or_tree if isinstance(sbend_pat_or_tree, Pattern) else sbend_pat_or_tree.top_pattern()
        dxy, _ = sbpat[sbend.in_port_name].measure_travel(sbpat[sbend.out_port_name])
        return dxy

    @staticmethod
    def _itransition2dxy(in_transition: Transition | None) -> NDArray[numpy.float64]:
        if in_transition is None:
            return numpy.zeros(2)
        dxy, _ = in_transition.their_port.measure_travel(in_transition.our_port)
        return dxy

    @staticmethod
    def _otransition2dxy(out_transition: Transition | None, bend_angle: float) -> NDArray[numpy.float64]:
        if out_transition is None:
            return numpy.zeros(2)
        orot = out_transition.our_port.rotation
        assert orot is not None
        otrans_dxy = rotation_matrix_2d(pi - orot - bend_angle) @ (out_transition.their_port.offset - out_transition.our_port.offset)
        return otrans_dxy

    def planL(
            self,
            ccw: SupportsBool | None,
            length: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[Port, LData]:

        success = False
        for straight in self.straights:
            for bend in self.bends:
                bend_dxy, bend_angle = self._bend2dxy(bend, ccw)

                in_ptype_pair = ('unk' if in_ptype is None else in_ptype, straight.ptype)
                in_transition = self.transitions.get(in_ptype_pair, None)
                itrans_dxy = self._itransition2dxy(in_transition)

                out_ptype_pair = (
                    'unk' if out_ptype is None else out_ptype,
                    straight.ptype if ccw is None else bend.out_port.ptype
                    )
                out_transition = self.transitions.get(out_ptype_pair, None)
                otrans_dxy = self._otransition2dxy(out_transition, bend_angle)

                b_transition = None
                if ccw is not None and bend.in_port.ptype != straight.ptype:
                    b_transition = self.transitions.get((bend.in_port.ptype, straight.ptype), None)
                btrans_dxy = self._itransition2dxy(b_transition)

                straight_length = length - bend_dxy[0] - itrans_dxy[0] - btrans_dxy[0] - otrans_dxy[0]
                bend_run = bend_dxy[1] + itrans_dxy[1] + btrans_dxy[1] + otrans_dxy[1]
                success = straight.length_range[0] <= straight_length < straight.length_range[1]
                if success:
                    break
            if success:
                break
        else:
            # Failed to break
            raise BuildError(
                f'Asked to draw L-path with total length {length:,g}, shorter than required bends and transitions:\n'
                f'bend: {bend_dxy[0]:,g}  in_trans: {itrans_dxy[0]:,g}\n'
                f'out_trans: {otrans_dxy[0]:,g} bend_trans: {btrans_dxy[0]:,g}'
                )

        if out_transition is not None:
            out_ptype_actual = out_transition.their_port.ptype
        elif ccw is not None:
            out_ptype_actual = bend.out_port.ptype
        elif not numpy.isclose(straight_length, 0):
            out_ptype_actual = straight.ptype
        else:
            out_ptype_actual = self.default_out_ptype

        data = self.LData(straight_length, straight, kwargs, ccw, bend, in_transition, b_transition, out_transition)
        out_port = Port((length, bend_run), rotation=bend_angle, ptype=out_ptype_actual)
        return out_port, data

    def _renderL(
            self,
            data: LData,
            tree: ILibrary,
            port_names: tuple[str, str],
            straight_kwargs: dict[str, Any],
            ) -> ILibrary:
        """
        Render an L step into a preexisting tree
        """
        pat = tree.top_pattern()
        if data.in_transition:
            pat.plug(data.in_transition.abstract, {port_names[1]: data.in_transition.their_port_name})
        if not numpy.isclose(data.straight_length, 0):
            straight_pat_or_tree = data.straight.fn(data.straight_length, **(straight_kwargs | data.straight_kwargs))
            pmap = {port_names[1]: data.straight.in_port_name}
            if isinstance(straight_pat_or_tree, Pattern):
                pat.plug(straight_pat_or_tree, pmap, append=True)
            else:
                straight_tree = straight_pat_or_tree
                top = straight_tree.top()
                straight_tree.flatten(top)
                pat.plug(straight_tree[top], pmap, append=True)
        if data.b_transition:
            pat.plug(data.b_transition.abstract, {port_names[1]: data.b_transition.our_port_name})
        if data.ccw is not None:
            assert data.bend is not None
            pat.plug(data.bend.abstract, {port_names[1]: data.bend.in_port_name}, mirrored=bool(data.ccw) == data.bend.clockwise)
        if data.out_transition:
            pat.plug(data.out_transition.abstract, {port_names[1]: data.out_transition.our_port_name})
        return tree

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
            in_ptype = in_ptype,
            out_ptype = out_ptype,
            )

        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
        pat.add_port_pair(names=port_names, ptype='unk' if in_ptype is None else in_ptype)
        self._renderL(data=data, tree=tree, port_names=port_names, straight_kwargs=kwargs)
        return tree

    def planS(
            self,
            length: float,
            jog: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            **kwargs,
            ) -> tuple[Port, Any]:

        success = False
        for straight in self.straights:
            for sbend in self.sbends:
                out_ptype_pair = (
                    'unk' if out_ptype is None else out_ptype,
                    straight.ptype if numpy.isclose(jog, 0) else sbend.ptype
                    )
                out_transition = self.transitions.get(out_ptype_pair, None)
                otrans_dxy = self._otransition2dxy(out_transition, pi)

                # Assume we'll need a straight segment with transitions, then discard them if they don't fit
                # We do this before generating the s-bend because the transitions might have some dy component
                in_ptype_pair = ('unk' if in_ptype is None else in_ptype, straight.ptype)
                in_transition = self.transitions.get(in_ptype_pair, None)
                itrans_dxy = self._itransition2dxy(in_transition)

                b_transition = None
                if not numpy.isclose(jog, 0) and sbend.ptype != straight.ptype:
                    b_transition = self.transitions.get((sbend.ptype, straight.ptype), None)
                btrans_dxy = self._itransition2dxy(b_transition)

                if length > itrans_dxy[0] + btrans_dxy[0] + otrans_dxy[0]:
                    # `if` guard to avoid unnecessary calls to `_sbend2dxy()`, which calls `sbend.fn()`
                    # note some S-bends may have 0 length, so we can't be more restrictive
                    jog_remaining = jog - itrans_dxy[1] - btrans_dxy[1] - otrans_dxy[1]
                    sbend_dxy = self._sbend2dxy(sbend, jog_remaining)
                    straight_length = length - sbend_dxy[0] - itrans_dxy[0] - btrans_dxy[0] - otrans_dxy[0]
                    success = straight.length_range[0] <= straight_length < straight.length_range[1]
                    if success:
                        break

                # Straight didn't work, see if just the s-bend is enough
                if sbend.ptype != straight.ptype:
                    # Need to use a different in-transition for sbend (vs straight)
                    in_ptype_pair = ('unk' if in_ptype is None else in_ptype, sbend.ptype)
                    in_transition = self.transitions.get(in_ptype_pair, None)
                    itrans_dxy = self._itransition2dxy(in_transition)

                jog_remaining = jog - itrans_dxy[1] - otrans_dxy[1]
                if sbend.jog_range[0] <= jog_remaining < sbend.jog_range[1]:
                    sbend_dxy = self._sbend2dxy(sbend, jog_remaining)
                    success = numpy.isclose(length, sbend_dxy[0] + itrans_dxy[1] + otrans_dxy[1])
                    if success:
                        b_transition = None
                        straight_length = 0
                        break
            if success:
                break

        if not success:
            try:
                ccw0 = jog > 0
                p_test0, ldata_test0 = self.planL(length / 2, ccw0, in_ptype=in_ptype)
                p_test1, ldata_test1 = self.planL(jog - p_test0.y, not ccw0, in_ptype=p_test0.ptype, out_ptype=out_ptype)

                dx = p_test1.x - length / 2
                p0, ldata0 = self.planL(length - dx, ccw0, in_ptype=in_ptype)
                p1, ldata1 = self.planL(jog - p0.y, not ccw0, in_ptype=p0.ptype, out_ptype=out_ptype)
                success = True
            except BuildError as err:
                l2_err: BuildError | None = err
            else:
                l2_err = None

        if not success:
            # Failed to break
            raise BuildError(
                f'Failed to find a valid s-bend configuration for {length=:,g}, {jog=:,g}, {in_ptype=}, {out_ptype=}'
                ) from l2_err

        if out_transition is not None:
            out_ptype_actual = out_transition.their_port.ptype
        elif not numpy.isclose(jog_remaining, 0):
            out_ptype_actual = sbend.ptype
        elif not numpy.isclose(straight_length, 0):
            out_ptype_actual = straight.ptype
        else:
            out_ptype_actual = self.default_out_ptype

        data = self.SData(straight_length, straight, kwargs, jog_remaining, sbend, in_transition, b_transition, out_transition)
        out_port = Port((length, jog), rotation=pi, ptype=out_ptype_actual)
        return out_port, data

    def _renderS(
            self,
            data: SData,
            tree: ILibrary,
            port_names: tuple[str, str],
            gen_kwargs: dict[str, Any],
            ) -> ILibrary:
        """
        Render an L step into a preexisting tree
        """
        pat = tree.top_pattern()
        if data.in_transition:
            pat.plug(data.in_transition.abstract, {port_names[1]: data.in_transition.their_port_name})
        if not numpy.isclose(data.straight_length, 0):
            straight_pat_or_tree = data.straight.fn(data.straight_length, **(gen_kwargs | data.gen_kwargs))
            pmap = {port_names[1]: data.straight.in_port_name}
            if isinstance(straight_pat_or_tree, Pattern):
                straight_pat = straight_pat_or_tree
                pat.plug(straight_pat, pmap, append=True)
            else:
                straight_tree = straight_pat_or_tree
                top = straight_tree.top()
                straight_tree.flatten(top)
                pat.plug(straight_tree[top], pmap, append=True)
        if data.b_transition:
            pat.plug(data.b_transition.abstract, {port_names[1]: data.b_transition.our_port_name})
        if not numpy.isclose(data.jog_remaining, 0):
            sbend_pat_or_tree = data.sbend.fn(abs(data.jog_remaining), **(gen_kwargs | data.gen_kwargs))
            pmap = {port_names[1]: data.sbend.in_port_name}
            if isinstance(sbend_pat_or_tree, Pattern):
                pat.plug(sbend_pat_or_tree, pmap, append=True, mirrored=data.jog_remaining < 0)
            else:
                sbend_tree = sbend_pat_or_tree
                top = sbend_tree.top()
                sbend_tree.flatten(top)
                pat.plug(sbend_tree[top], pmap, append=True, mirrored=data.jog_remaining < 0)
        if data.out_transition:
            pat.plug(data.out_transition.abstract, {port_names[1]: data.out_transition.our_port_name})
        return tree

    def pathS(
            self,
            length: float,
            jog: float,
            *,
            in_ptype: str | None = None,
            out_ptype: str | None = None,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> Library:
        _out_port, data = self.planS(
            length,
            jog,
            in_ptype = in_ptype,
            out_ptype = out_ptype,
            )
        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'pathS')
        pat.add_port_pair(names=port_names, ptype='unk' if in_ptype is None else in_ptype)
        self._renderS(data=data, tree=tree, port_names=port_names, gen_kwargs=kwargs)
        return tree

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,
            ) -> ILibrary:

        tree, pat = Library.mktree(SINGLE_USE_PREFIX + 'path')
        pat.add_port_pair(names=(port_names[0], port_names[1]))

        for step in batch:
            assert step.tool == self
            if step.opcode == 'L':
                self._renderL(data=step.data, tree=tree, port_names=port_names, straight_kwargs=kwargs)
            elif step.opcode == 'S':
                self._renderS(data=step.data, tree=tree, port_names=port_names, gen_kwargs=kwargs)
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
            **kwargs,                           # noqa: ARG002 (unused)
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
            in_ptype: str | None = None,        # noqa: ARG002 (unused)
            out_ptype: str | None = None,
            **kwargs,                           # noqa: ARG002 (unused)
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
                f'Asked to draw L-path with total length {length:,g}, shorter than required bend: {bend_dxy[0]:,g}'
                )
        data = numpy.array((length, bend_run))
        out_port = Port(data, rotation=bend_angle, ptype=self.ptype)
        return out_port, data

    def render(
            self,
            batch: Sequence[RenderStep],
            *,
            port_names: tuple[str, str] = ('A', 'B'),
            **kwargs,                           # noqa: ARG002 (unused)
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
