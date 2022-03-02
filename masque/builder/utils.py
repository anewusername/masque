from typing import Dict, Tuple, List, Optional, Union, Any, cast, Sequence, TYPE_CHECKING
from pprint import pformat

import numpy
from numpy import pi
from numpy.typing import ArrayLike

from ..utils import rotation_matrix_2d
from ..error import BuildError

if TYPE_CHECKING:
    from .devices import Port


def ell(
        ports: Dict[str, 'Port'],
        ccw: Optional[bool],
        bound_type: str,
        bound: Union[float, ArrayLike],
        *,
        spacing: Optional[Union[float, ArrayLike]] = None,
        set_rotation: Optional[float] = None,
        ) -> Dict[str, float]:
    """
    Calculate extension for each port in order to build a 90-degree bend with the provided
    channel spacing:

         =A>---------------------------V     turn direction: `ccw=False`
                   =B>-------------V   |
     =C>-----------------------V   |   |
       =D=>----------------V   |   |   |


                           x---x---x---x  `spacing` (can be scalar or array)

                    <-------------->      `bound_type='min_extension'`
                    <------>                         `'min_past_furthest'`
      <-------------------------------->             `'max_extension'`
                           x                         `'min_position'`
                                       x             `'max_position'`

    Args:
        ports: `name: port` mapping. All ports should have the same rotation (or `None`). If
            no port has a rotation specified, `set_rotation` must be provided.
        ccw: Turn direction. `True` means counterclockwise, `False` means clockwise,
            and `None` means no bend. If `None`, spacing must remain `None` or `0` (default),
            Otherwise, spacing must be set to a non-`None` value.
        bound_method: Method used for determining the travel distance; see diagram above.
            Valid values are:
            - 'min_extension' or 'emin':
                The total extension value for the furthest-out port (B in the diagram).
            - 'min_past_furthest':
                The distance between furthest out-port (B) and the innermost bend (D's bend).
            - 'max_extension' or 'emax':
                The total extension value for the closest-in port (C in the diagram).
            - 'min_position' or 'pmin':
                The coordinate of the innermost bend (D's bend).
            - 'max_position' or 'pmax':
                The coordinate of the outermost bend (A's bend).

            `bound` can also be a vector. If specifying an extension (e.g. 'min_extension',
                'max_extension', 'min_past_furthest'), it sets independent limits along
                the x- and y- axes. If specifying a position, it is projected onto
                the extension direction.

        bound_value: Value associated with `bound_type`, see above.
        spacing: Distance between adjacent channels. Can be scalar, resulting in evenly
            spaced channels, or a vector with length one less than `ports`, allowing
            non-uniform spacing.
            The ordering of the vector corresponds to the output order (DCBA in the
            diagram above), *not* the order of `ports`.
        set_rotation: If all ports have no specified rotation, this value is used
            to set the extension direction. Otherwise it must remain `None`.

    Returns:
        Dict of {port_name: distance_to_bend}

    Raises:
        `BuildError` on bad inputs
        `BuildError` if the requested bound is impossible
    """
    if not ports:
        raise BuildError('Empty port list passed to `ell()`')

    if ccw is None:
        if spacing is not None and not numpy.isclose(spacing, 0):
            raise BuildError('Spacing must be 0 or None when ccw=None')
        spacing = 0
    elif spacing is None:
        raise BuildError('Must provide spacing if a bend direction is specified')

    has_rotation = numpy.array([p.rotation is not None for p in ports.values()], dtype=bool)
    if has_rotation.any():
        if set_rotation is not None:
            raise BuildError('set_rotation must be None when ports have rotations!')

        rotations = numpy.array([p.rotation if p.rotation is not None else 0
                                 for p in ports.values()])
        rotations[~has_rotation] = rotations[has_rotation][0]

        if not numpy.allclose(rotations[0], rotations):
            port_rotations = {k: numpy.rad2deg(p.rotation) if p.rotation is not None else None
                              for k, p in ports.items()}

            raise BuildError('Asked to find aggregation for ports that face in different directions:\n'
                             + pformat(port_rotations))
    else:
        if set_rotation is not None:
            raise BuildError('set_rotation must be specified if no ports have rotations!')
        rotations = numpy.full_like(has_rotation, set_rotation, dtype=float)

    direction = rotations[0] + pi                        # direction we want to travel in (+pi relative to port)
    rot_matrix = rotation_matrix_2d(-direction)

    # Rotate so are traveling in +x
    orig_offsets = numpy.array([p.offset for p in ports.values()])
    rot_offsets = (rot_matrix @ orig_offsets.T).T

    y_order = ((-1 if ccw else 1) * rot_offsets[:, 1]).argsort(kind='stable')
    y_ind = numpy.empty_like(y_order, dtype=int)
    y_ind[y_order] = numpy.arange(y_ind.shape[0])

    if spacing is None:
        ch_offsets = numpy.zeros_like(y_order)
    else:
        steps = numpy.zeros_like(y_order)
        steps[1:] = spacing
        ch_offsets = numpy.cumsum(steps)[y_ind]

    x_start = rot_offsets[:, 0]

    #     A---------|  `d_to_align[0]`
    #               B  `d_to_align[1]`
    # C-------------|  `d_to_align[2]`
    #   D-----------|  `d_to_align[3]`
    #
    d_to_align = x_start.max() - x_start    # distance to travel to align all
    if bound_type == 'min_past_furthest':
        #     A------------------V  `d_to_exit[0]`
        #               B-----V     `d_to_exit[1]`
        # C----------------V        `d_to_exit[2]`
        #   D-----------V           `d_to_exit[3]`
        offsets = d_to_align + ch_offsets
    else:
        #     A---------V  `travel[0]`   <-- Outermost port aligned to furthest-x port
        #            V--B  `travel[1]`   <-- Remaining ports follow spacing
        # C-------V        `travel[2]`
        #   D--V           `travel[3]`
        #
        #     A------------V  `offsets[0]`
        #               B     `offsets[1]`   <-- Travels adjusted to be non-negative
        # C----------V        `offsets[2]`
        #   D-----V           `offsets[3]`
        travel = d_to_align - (ch_offsets.max() - ch_offsets)
        offsets = travel - travel.min().clip(max=0)

    if bound_type in ('emin', 'min_extension',
                      'emax', 'max_extension',
                      'min_past_furthest',):
        if numpy.size(bound) == 2:
            bound = cast(Sequence[float], bound)
            rot_bound = (rot_matrix @ ((bound[0], 0),
                                       (0, bound[1])))[0, :]
        else:
            bound = cast(float, bound)
            rot_bound = numpy.array(bound)

        if rot_bound < 0:
            raise BuildError(f'Got negative bound for extension: {rot_bound}')

        if bound_type in ('emin', 'min_extension', 'min_past_furthest'):
            offsets += rot_bound.max()
        elif bound_type in('emax', 'max_extension'):
            offsets += rot_bound.min() - offsets.max()
    else:
        if numpy.size(bound) == 2:
            bound = cast(Sequence[float], bound)
            rot_bound = (rot_matrix @ bound)[0]
        else:
            bound = cast(float, bound)
            neg = (direction + pi / 4) % (2 * pi) > pi
            rot_bound = -bound if neg else bound

        min_possible = x_start + offsets
        if bound_type in ('pmax', 'max_position'):
            extension = rot_bound - min_possible.max()
        elif bound_type in ('pmin', 'min_position'):
            extension = rot_bound - min_possible.min()

        offsets += extension
        if extension < 0:
            raise BuildError(f'Position is too close by at least {-numpy.floor(extension)}. Total extensions would be'
                            + '\n\t'.join(f'{key}: {off}' for key, off in zip(ports.keys(), offsets)))

    result = dict(zip(ports.keys(), offsets))
    return result
