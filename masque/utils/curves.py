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
    nn = nodes.shape[0]
    if weights is None:
        weights = numpy.ones(nn)

    t_half0 = tt <= 0.5
    umul = tt / (1 - tt)
    udiv = 1 / umul
    umul[~t_half0] = 1
    udiv[t_half0] = 1

    hh = numpy.ones((tt.size, 1))
    qq = nodes[None, 0] * hh
    for kk in range(1, nn):
        hh *= umul * (nn + 1 - kk) * weights[kk]
        hh /= kk * udiv * weights[kk - 1] + hh
        qq *= 1.0 - hh
        qq += hh * nodes[None, kk]
    return qq



def euler_bend(switchover_angle: float) -> NDArray[numpy.float64]:
    """
    Generate a 90 degree Euler bend (AKA Clothoid bend or Cornu spiral).

    Args:
        switchover_angle: After this angle, the bend will transition into a circular arc
            (and transition back to an Euler spiral on the far side). If this is set to
            `>= pi / 4`, no circular arc will be added.

    Returns:
        `[[x0, y0], ...]` for the curve
    """
    # Switchover angle
    # AKA: Clothoid bend, Cornu spiral
    theta_max = numpy.sqrt(2 * switchover_angle)

    def gen_curve(theta_max: float):
        xx = []
        yy = []
        for theta in numpy.linspace(0, theta_max, 100):
            qq = numpy.linspace(0, theta, 1000)
            xx.append(numpy.trapz( numpy.cos(qq * qq / 2), qq))
            yy.append(numpy.trapz(-numpy.sin(qq * qq / 2), qq))
        xy_part = numpy.stack((xx, yy), axis=1)
        return xy_part

    xy_part = gen_curve(theta_max)
    xy_parts = [xy_part]

    if switchover_angle < pi / 4:
        # Build a circular segment to join the two euler portions
        rmin = 1.0 / theta_max
        half_angle = pi / 4 - switchover_angle
        qq = numpy.linspace(half_angle * 2, 0, 10) + switchover_angle
        xc = rmin * numpy.cos(qq)
        yc = rmin * numpy.sin(qq) + xy_part[-1, 1]
        xc += xy_part[-1, 0] - xc[0]
        yc += xy_part[-1, 1] - yc[0]
        xy_parts.append(numpy.stack((xc, yc), axis=1))

    endpoint_xy = xy_parts[-1][-1, :]
    second_curve = xy_part[::-1, ::-1] + endpoint_xy - xy_part[-1, ::-1]

    xy_parts.append(second_curve)
    xy = numpy.concatenate(xy_parts)

    # Remove any 2x-duplicate points
    xy = xy[(numpy.roll(xy, 1, axis=0) != xy).any(axis=1)]

    return xy
