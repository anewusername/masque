import numpy
from numpy.typing import ArrayLike, NDArray
from numpy import pi


def bezier(
        nodes: ArrayLike,
        tt: ArrayLike,
        weights: ArrayLike | None = None,
        ) -> NDArray[numpy.float64]:
    """
    Sample a Bezier curve with the provided control points at the parametrized positions `tt`.

    Using the calculation method from arXiv:1803.06843, Chudy and Woźny.

    Args:
        nodes: `[[x0, y0], ...]` control points for the Bezier curve
        tt: Parametrized positions at which to sample the curve (1D array with values in the interval [0, 1])
        weights: Control point weights; if provided, length should be the same as number of control points.
            Default 1 for all control points.

    Returns:
        `[[x0, y0], [x1, y1], ...]` corresponding to `[tt0, tt1, ...]`
    """
    nodes = numpy.asarray(nodes)
    tt = numpy.asarray(tt)
    nn = nodes.shape[0]
    weights = numpy.ones(nn) if weights is None else numpy.asarray(weights)

    with numpy.errstate(divide='ignore'):
        umul = (tt / (1 - tt)).clip(max=1)
        udiv = ((1 - tt) / tt).clip(max=1)

    hh = numpy.ones((tt.size,))
    qq = nodes[None, 0, :] * hh[:, None]
    for kk in range(1, nn):
        hh *= umul * (nn - kk) * weights[kk]
        hh /= kk * udiv * weights[kk - 1] + hh
        qq *= 1.0 - hh[:, None]
        qq += hh[:, None] * nodes[None, kk, :]
    return qq



def euler_bend(
        switchover_angle: float,
        num_points: int = 200,
        ) -> NDArray[numpy.float64]:
    """
    Generate a 90 degree Euler bend (AKA Clothoid bend or Cornu spiral).

    Args:
        switchover_angle: After this angle, the bend will transition into a circular arc
            (and transition back to an Euler spiral on the far side). If this is set to
            `>= pi / 4`, no circular arc will be added.
        num_points: Number of points in the curve

    Returns:
        `[[x0, y0], ...]` for the curve
    """
    ll_max = numpy.sqrt(2 * switchover_angle)        # total length of (one) spiral portion
    ll_tot = 2 * ll_max + (pi / 2 - 2 * switchover_angle)
    num_points_spiral = numpy.floor(ll_max / ll_tot * num_points).astype(int)
    num_points_arc = num_points - 2 * num_points_spiral

    def gen_spiral(ll_max: float):
        xx = []
        yy = []
        for ll in numpy.linspace(0, ll_max, num_points_spiral):
            qq = numpy.linspace(0, ll, 1000)        # integrate to current arclength
            xx.append(numpy.trapz( numpy.cos(qq * qq / 2), qq))
            yy.append(numpy.trapz(-numpy.sin(qq * qq / 2), qq))
        xy_part = numpy.stack((xx, yy), axis=1)
        return xy_part

    xy_spiral = gen_spiral(ll_max)
    xy_parts = [xy_spiral]

    if switchover_angle < pi / 4:
        # Build a circular segment to join the two euler portions
        rmin = 1.0 / ll_max
        half_angle = pi / 4 - switchover_angle
        qq = numpy.linspace(half_angle * 2, 0, num_points_arc + 1) + switchover_angle
        xc = rmin * numpy.cos(qq)
        yc = rmin * numpy.sin(qq) + xy_spiral[-1, 1]
        xc += xy_spiral[-1, 0] - xc[0]
        yc += xy_spiral[-1, 1] - yc[0]
        xy_parts.append(numpy.stack((xc[1:], yc[1:]), axis=1))

    endpoint_xy = xy_parts[-1][-1, :]
    second_spiral = xy_spiral[::-1, ::-1] + endpoint_xy - xy_spiral[-1, ::-1]

    xy_parts.append(second_spiral)
    xy = numpy.concatenate(xy_parts)

    # Remove any 2x-duplicate points
    xy = xy[(numpy.roll(xy, 1, axis=0) != xy).any(axis=1)]

    return xy
