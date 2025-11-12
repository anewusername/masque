from typing import Self
from collections.abc import Sequence, Iterator
import logging
from contextlib import contextmanager
from abc import abstractmethod, ABCMeta

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from ..pattern import Pattern
from ..library import ILibrary
from ..error import PortError, BuildError
from ..utils import rotation_matrix_2d, SupportsBool
#from ..abstract import Abstract
from .tools import Tool
from .utils import ell


logger = logging.getLogger(__name__)


class PatherMixin(metaclass=ABCMeta):
    pattern: Pattern
    """ Layout of this device """

    library: ILibrary
    """ Library from which patterns should be referenced """

    _dead: bool
    """ If True, plug()/place() are skipped (for debugging) """

    tools: dict[str | None, Tool]
    """
    Tool objects are used to dynamically generate new single-use Devices
    (e.g wires or waveguides) to be plugged into this device.
    """

    @abstractmethod
    def path(
            self,
            portspec: str,
            ccw: SupportsBool | None,
            length: float,
            *,
            plug_into: str | None = None,
            **kwargs,
            ) -> Self:
        pass

    def retool(
            self,
            tool: Tool,
            keys: str | Sequence[str | None] | None = None,
            ) -> Self:
        """
        Update the `Tool` which will be used when generating `Pattern`s for the ports
        given by `keys`.

        Args:
            tool: The new `Tool` to use for the given ports.
            keys: Which ports the tool should apply to. `None` indicates the default tool,
                used when there is no matching entry in `self.tools` for the port in question.

        Returns:
            self
        """
        if keys is None or isinstance(keys, str):
            self.tools[keys] = tool
        else:
            for key in keys:
                self.tools[key] = tool
        return self

    @contextmanager
    def toolctx(
            self,
            tool: Tool,
            keys: str | Sequence[str | None] | None = None,
            ) -> Iterator[Self]:
        """
          Context manager for temporarily `retool`-ing and reverting the `retool`
        upon exiting the context.

        Args:
            tool: The new `Tool` to use for the given ports.
            keys: Which ports the tool should apply to. `None` indicates the default tool,
                used when there is no matching entry in `self.tools` for the port in question.

        Returns:
            self
        """
        if keys is None or isinstance(keys, str):
            keys = [keys]
        saved_tools = {kk: self.tools.get(kk, None) for kk in keys}      # If not in self.tools, save `None`
        try:
            yield self.retool(tool=tool, keys=keys)
        finally:
            for kk, tt in saved_tools.items():
                if tt is None:
                    # delete if present
                    self.tools.pop(kk, None)
                else:
                    self.tools[kk] = tt

    def path_to(
            self,
            portspec: str,
            ccw: SupportsBool | None,
            position: float | None = None,
            *,
            x: float | None = None,
            y: float | None = None,
            plug_into: str | None = None,
            **kwargs,
            ) -> Self:
        """
        Build  a "wire"/"waveguide" extending from the port `portspec`, with the aim
        of ending exactly at a target position.

        The wire will travel so that the output port will be placed at exactly the target
        position along the input port's axis. There can be an unspecified (tool-dependent)
        offset in the perpendicular direction. The output port will be rotated (or not)
        based on the `ccw` parameter.

        If using `RenderPather`, `RenderPather.render` must be called after all paths have been fully planned.

        Args:
            portspec: The name of the port into which the wire will be plugged.
            ccw: If `None`, the output should be along the same axis as the input.
                Otherwise, cast to bool and turn counterclockwise if True
                and clockwise otherwise.
            position: The final port position, along the input's axis only.
                (There may be a tool-dependent offset along the other axis.)
                Only one of `position`, `x`, and `y` may be specified.
            x: The final port position along the x axis.
                `portspec` must refer to a horizontal port if `x` is passed, otherwise a
                BuildError will be raised.
            y: The final port position along the y axis.
                `portspec` must refer to a vertical port if `y` is passed, otherwise a
                BuildError will be raised.
            plug_into: If not None, attempts to plug the wire's output port into the provided
                port on `self`.

        Returns:
            self

        Raises:
            BuildError if `position`, `x`, or `y` is too close to fit the bend (if a bend
                is present).
            BuildError if `x` or `y` is specified but does not match the axis of `portspec`.
            BuildError if more than one of `x`, `y`, and `position` is specified.
        """
        if self._dead:
            logger.error('Skipping path_to() since device is dead')
            return self

        pos_count = sum(vv is not None for vv in (position, x, y))
        if pos_count > 1:
            raise BuildError('Only one of `position`, `x`, and `y` may be specified at once')
        if pos_count < 1:
            raise BuildError('One of `position`, `x`, and `y` must be specified')

        port = self.pattern[portspec]
        if port.rotation is None:
            raise PortError(f'Port {portspec} has no rotation and cannot be used for path_to()')

        if not numpy.isclose(port.rotation % (pi / 2), 0):
            raise BuildError('path_to was asked to route from non-manhattan port')

        is_horizontal = numpy.isclose(port.rotation % pi, 0)
        if is_horizontal:
            if y is not None:
                raise BuildError('Asked to path to y-coordinate, but port is horizontal')
            if position is None:
                position = x
        else:
            if x is not None:
                raise BuildError('Asked to path to x-coordinate, but port is vertical')
            if position is None:
                position = y

        x0, y0 = port.offset
        if is_horizontal:
            if numpy.sign(numpy.cos(port.rotation)) == numpy.sign(position - x0):
                raise BuildError(f'path_to routing to behind source port: x0={x0:g} to {position:g}')
            length = numpy.abs(position - x0)
        else:
            if numpy.sign(numpy.sin(port.rotation)) == numpy.sign(position - y0):
                raise BuildError(f'path_to routing to behind source port: y0={y0:g} to {position:g}')
            length = numpy.abs(position - y0)

        return self.path(
            portspec,
            ccw,
            length,
            plug_into = plug_into,
            **kwargs,
            )

    def path_into(
            self,
            portspec_src: str,
            portspec_dst: str,
            *,
            out_ptype: str | None = None,
            plug_destination: bool = True,
            **kwargs,
            ) -> Self:
        """
        Create a "wire"/"waveguide" traveling between the ports `portspec_src` and
        `portspec_dst`, and `plug` it into both (or just the source port).

        Only unambiguous scenarios are allowed:
            - Straight connector between facing ports
            - Single 90 degree bend
            - Jog between facing ports
                (jog is done as late as possible, i.e. only 2 L-shaped segments are used)

        By default, the destination's `pytpe` will be used as the `out_ptype` for the
        wire, and the `portspec_dst` will be plugged (i.e. removed).

        If using `RenderPather`, `RenderPather.render` must be called after all paths have been fully planned.

        Args:
            portspec_src: The name of the starting port into which the wire will be plugged.
            portspec_dst: The name of the destination port.
            out_ptype: Passed to the pathing tool in order to specify the desired port type
                to be generated at the destination end. If `None` (default), the destination
                port's `ptype` will be used.

        Returns:
            self

        Raises:
            PortError if either port does not have a specified rotation.
            BuildError if and invalid port config is encountered:
                - Non-manhattan ports
                - U-bend
                - Destination too close to (or behind) source
        """
        if self._dead:
            logger.error('Skipping path_into() since device is dead')
            return self

        port_src = self.pattern[portspec_src]
        port_dst = self.pattern[portspec_dst]

        if out_ptype is None:
            out_ptype = port_dst.ptype

        if port_src.rotation is None:
            raise PortError(f'Port {portspec_src} has no rotation and cannot be used for path_into()')
        if port_dst.rotation is None:
            raise PortError(f'Port {portspec_dst} has no rotation and cannot be used for path_into()')

        if not numpy.isclose(port_src.rotation % (pi / 2), 0):
            raise BuildError('path_into was asked to route from non-manhattan port')
        if not numpy.isclose(port_dst.rotation % (pi / 2), 0):
            raise BuildError('path_into was asked to route to non-manhattan port')

        src_is_horizontal = numpy.isclose(port_src.rotation % pi, 0)
        dst_is_horizontal = numpy.isclose(port_dst.rotation % pi, 0)
        xs, ys = port_src.offset
        xd, yd = port_dst.offset

        angle = (port_dst.rotation - port_src.rotation) % (2 * pi)

        src_ne = port_src.rotation % (2 * pi) > (3 * pi / 4)     # path from src will go north or east

        def get_jog(ccw: SupportsBool, length: float) -> float:
            tool = self.tools.get(portspec_src, self.tools[None])
            in_ptype = 'unk'   # Could use port_src.ptype, but we're assuming this is after one bend already...
            tree2 = tool.path(ccw, length, in_ptype=in_ptype, port_names=('A', 'B'), out_ptype=out_ptype, **kwargs)
            top2 = tree2.top_pattern()
            jog = rotation_matrix_2d(top2['A'].rotation) @ (top2['B'].offset - top2['A'].offset)
            return jog[1] * [-1, 1][int(bool(ccw))]

        dst_extra_args = {'out_ptype': out_ptype}
        if plug_destination:
            dst_extra_args['plug_into'] = portspec_dst

        src_args = {**kwargs}
        dst_args = {**src_args, **dst_extra_args}
        if src_is_horizontal and not dst_is_horizontal:
            # single bend should suffice
            self.path_to(portspec_src, angle > pi, x=xd, **src_args)
            self.path_to(portspec_src, None, y=yd, **dst_args)
        elif dst_is_horizontal and not src_is_horizontal:
            # single bend should suffice
            self.path_to(portspec_src, angle > pi, y=yd, **src_args)
            self.path_to(portspec_src, None, x=xd, **dst_args)
        elif numpy.isclose(angle, pi):
            if src_is_horizontal and ys == yd:
                # straight connector
                self.path_to(portspec_src, None, x=xd, **dst_args)
            elif not src_is_horizontal and xs == xd:
                # straight connector
                self.path_to(portspec_src, None, y=yd, **dst_args)
            elif src_is_horizontal:
                # figure out how much x our y-segment (2nd) takes up, then path based on that
                y_len = numpy.abs(yd - ys)
                ccw2 = src_ne != (yd > ys)
                jog = get_jog(ccw2, y_len) * numpy.sign(xd - xs)
                self.path_to(portspec_src, not ccw2, x=xd - jog, **src_args)
                self.path_to(portspec_src, ccw2, y=yd, **dst_args)
            else:
                # figure out how much y our x-segment (2nd) takes up, then path based on that
                x_len = numpy.abs(xd - xs)
                ccw2 = src_ne != (xd < xs)
                jog = get_jog(ccw2, x_len) * numpy.sign(yd - ys)
                self.path_to(portspec_src, not ccw2, y=yd - jog, **src_args)
                self.path_to(portspec_src, ccw2, x=xd, **dst_args)
        elif numpy.isclose(angle, 0):
            raise BuildError('Don\'t know how to route a U-bend yet (TODO)!')
        else:
            raise BuildError(f'Don\'t know how to route ports with relative angle {angle}')

        return self

    def mpath(
            self,
            portspec: str | Sequence[str],
            ccw: SupportsBool | None,
            *,
            spacing: float | ArrayLike | None = None,
            set_rotation: float | None = None,
            **kwargs,
            ) -> Self:
        """
        `mpath` is a superset of `path` and `path_to` which can act on bundles or buses
        of "wires or "waveguides".

        The wires will travel so that the output ports will be placed at well-defined
        locations along the axis of their input ports, but may have arbitrary (tool-
        dependent) offsets in the perpendicular direction.

        If `ccw` is not `None`, the wire bundle will turn 90 degres in either the
        clockwise (`ccw=False`) or counter-clockwise (`ccw=True`) direction. Within the
        bundle, the center-to-center wire spacings after the turn are set by `spacing`,
        which is required when `ccw` is not `None`. The final position of bundle as a
        whole can be set in a number of ways:

             =A>---------------------------V     turn direction: `ccw=False`
                       =B>-------------V   |
         =C>-----------------------V   |
           =D=>----------------V   |
                               |

                               x---x---x---x  `spacing` (can be scalar or array)

                        <-------------->      `emin=`
                        <------>              `bound_type='min_past_furthest', bound=`
          <-------------------------------->  `emax=`
                               x              `pmin=`
                                           x  `pmax=`

            - `emin=`, equivalent to `bound_type='min_extension', bound=`
                The total extension value for the furthest-out port (B in the diagram).
            - `emax=`, equivalent to `bound_type='max_extension', bound=`:
                The total extension value for the closest-in port (C in the diagram).
            - `pmin=`, equivalent to `xmin=`, `ymin=`, or `bound_type='min_position', bound=`:
                The coordinate of the innermost bend (D's bend).
                The x/y versions throw an error if they do not match the port axis (for debug)
            - `pmax=`, `xmax=`, `ymax=`, or `bound_type='max_position', bound=`:
                The coordinate of the outermost bend (A's bend).
                The x/y versions throw an error if they do not match the port axis (for debug)
            - `bound_type='min_past_furthest', bound=`:
                The distance between furthest out-port (B) and the innermost bend (D's bend).

        If `ccw=None`, final output positions (along the input axis) of all wires will be
        identical (i.e. wires will all be cut off evenly). In this case, `spacing=None` is
        required. In this case, `emin=` and `emax=` are equivalent to each other, and
        `pmin=`, `pmax=`, `xmin=`, etc. are also equivalent to each other.

        If using `RenderPather`, `RenderPather.render` must be called after all paths have been fully planned.

        Args:
            portspec: The names of the ports which are to be routed.
            ccw: If `None`, the outputs should be along the same axis as the inputs.
                Otherwise, cast to bool and turn 90 degrees counterclockwise if `True`
                and clockwise otherwise.
            spacing: Center-to-center distance between output ports along the input port's axis.
                Must be provided if (and only if) `ccw` is not `None`.
            set_rotation: If the provided ports have `rotation=None`, this can be used
                to set a rotation for them.

        Returns:
            self

        Raises:
            BuildError if the implied length for any wire is too close to fit the bend
                (if a bend is requested).
            BuildError if `xmin`/`xmax` or `ymin`/`ymax` is specified but does not
                match the axis of `portspec`.
            BuildError if an incorrect bound type or spacing is specified.
        """
        if self._dead:
            logger.error('Skipping mpath() since device is dead')
            return self

        bound_types = set()
        if 'bound_type' in kwargs:
            bound_types.add(kwargs.pop('bound_type'))
            bound = kwargs.pop('bound')
        for bt in ('emin', 'emax', 'pmin', 'pmax', 'xmin', 'xmax', 'ymin', 'ymax', 'min_past_furthest'):
            if bt in kwargs:
                bound_types.add(bt)
                bound = kwargs.pop(bt)

        if not bound_types:
            raise BuildError('No bound type specified for mpath')
        if len(bound_types) > 1:
            raise BuildError(f'Too many bound types specified for mpath: {bound_types}')
        bound_type = tuple(bound_types)[0]

        if isinstance(portspec, str):
            portspec = [portspec]
        ports = self.pattern[tuple(portspec)]

        extensions = ell(ports, ccw, spacing=spacing, bound=bound, bound_type=bound_type, set_rotation=set_rotation)

        #if container:
        #    assert not getattr(self, 'render'), 'Containers not implemented for RenderPather'
        #    bld = self.interface(source=ports, library=self.library, tools=self.tools)
        #    for port_name, length in extensions.items():
        #        bld.path(port_name, ccw, length, **kwargs)
        #    self.library[container] = bld.pattern
        #    self.plug(Abstract(container, bld.pattern.ports), {sp: 'in_' + sp for sp in ports})       # TODO safe to use 'in_'?
        #else:
        for port_name, length in extensions.items():
            self.path(port_name, ccw, length, **kwargs)
        return self

    # TODO def bus_join()?

    def flatten(self) -> Self:
        """
        Flatten the contained pattern, using the contained library to resolve references.

        Returns:
            self
        """
        self.pattern.flatten(self.library)
        return self
