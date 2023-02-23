"""
Routines for creating normalized 2D lattices and common photonic crystal
 cavity designs.
"""
from typing import Sequence

import numpy
from numpy.typing import ArrayLike, NDArray


def triangular_lattice(
        dims: Sequence[int],
        asymmetric: bool = False,
        origin: str = 'center',
        ) -> NDArray[numpy.float64]:
    """
    Return an ndarray of `[[x0, y0], [x1, y1], ...]` denoting lattice sites for
     a triangular lattice in 2D.

    Args:
        dims: Number of lattice sites in the [x, y] directions.
        asymmetric: If true, each row will contain the same number of
            x-coord lattice sites. If false, every other row will be
            one site shorter (to make the structure symmetric).
        origin: If 'corner', the least-(x,y) lattice site is placed at (0, 0)
            If 'center', the center of the lattice (not necessarily a
            lattice site) is placed at (0, 0).

    Returns:
        `[[x0, y0], [x1, 1], ...]` denoting lattice sites.
    """
    sx, sy = numpy.meshgrid(
        numpy.arange(dims[0], dtype=float),
        numpy.arange(dims[1], dtype=float),
        indexing='ij',
        )

    sx[sy % 2 == 1] += 0.5
    sy *= numpy.sqrt(3) / 2

    if not asymmetric:
        which = sx != sx.max()
        sx = sx[which]
        sy = sy[which]

    xy = numpy.column_stack((sx.flat, sy.flat))

    if origin == 'center':
        xy -= (xy.max(axis=0) - xy.min(axis=0)) / 2
    elif origin == 'corner':
        pass
    else:
        raise Exception(f'Invalid value for `origin`: {origin}')

    return xy[xy[:, 0].argsort(), :]


def square_lattice(dims: Sequence[int]) -> NDArray[numpy.float64]:
    """
    Return an ndarray of `[[x0, y0], [x1, y1], ...]` denoting lattice sites for
     a square lattice in 2D. The lattice will be centered around (0, 0).

    Args:
        dims: Number of lattice sites in the [x, y] directions.

    Returns:
        `[[x0, y0], [x1, 1], ...]` denoting lattice sites.
    """
    xs, ys = numpy.meshgrid(range(dims[0]), range(dims[1]), 'xy')
    xs -= dims[0]/2
    ys -= dims[1]/2
    xy = numpy.vstack((xs.flatten(), ys.flatten())).T
    return xy[xy[:, 0].argsort(), ]


# ### Photonic crystal functions ###


def nanobeam_holes(
        a_defect: float,
        num_defect_holes: int,
        num_mirror_holes: int
        ) -> NDArray[numpy.float64]:
    """
    Returns a list of `[[x0, r0], [x1, r1], ...]` of nanobeam hole positions and radii.
     Creates a region in which the lattice constant and radius are progressively
     (linearly) altered over num_defect_holes holes until they reach the value
     specified by a_defect, then symmetrically returned to a lattice constant and
     radius of 1, which is repeated num_mirror_holes times on each side.

    Args:
        a_defect: Minimum lattice constant for the defect, as a fraction of the
            mirror lattice constant (ie., for no defect, a_defect = 1).
        num_defect_holes: How many holes form the defect (per-side)
        num_mirror_holes: How many holes form the mirror (per-side)

    Returns:
        Ndarray `[[x0, r0], [x1, r1], ...]` of nanobeam hole positions and radii.
    """
    a_values = numpy.linspace(a_defect, 1, num_defect_holes, endpoint=False)
    xs = a_values.cumsum() - (a_values[0] / 2)  # Later mirroring makes center distance 2x as long
    mirror_xs = numpy.arange(1, num_mirror_holes + 1, dtype=float) + xs[-1]
    mirror_rs = numpy.ones_like(mirror_xs)
    return numpy.vstack((numpy.hstack((-mirror_xs[::-1], -xs[::-1], xs, mirror_xs)),
                         numpy.hstack((mirror_rs[::-1], a_values[::-1], a_values, mirror_rs)))).T


def waveguide(length: int, num_mirror: int) -> NDArray[numpy.float64]:
    """
    Line defect waveguide in a triangular lattice.

    Args:
        length: waveguide length (number of holes in x direction)
        num_mirror: Mirror length (number of holes per side; total size is
            `2 * n + 1` holes.

    Returns:
        `[[x0, y0], [x1, y1], ...]` for all the holes
    """
    p = triangular_lattice([length + 2, 2 * num_mirror + 1])
    p = p[p[:, 1] != 0, :]

    p = p[numpy.abs(p[:, 0]) <= length / 2]
    return p


def wgbend(num_mirror: int) -> NDArray[numpy.float64]:
    """
    Line defect waveguide bend in a triangular lattice.

    Args:
        num_mirror: Mirror length (number of holes per side; total size is
            approximately `2 * n + 1`

    Returns:
        `[[x0, y0], [x1, y1], ...]` for all the holes
    """
    p = triangular_lattice([4 * num_mirror + 1, 4 * num_mirror + 1])
    left_horiz = (p[:, 1] == 0) & (p[:, 0] <= 0)
    p = p[~left_horiz, :]

    right_diag = numpy.isclose(p[:, 1], p[:, 0] * numpy.sqrt(3)) & (p[:, 0] >= 0)
    p = p[~right_diag, :]

    edge_left = p[:, 0] < -num_mirror
    edge_bot = p[:, 1] < -num_mirror
    p = p[~edge_left & ~edge_bot, :]

    edge_diag_up = p[:, 0] * numpy.sqrt(3) > p[:, 1] + 2 * num_mirror + 0.1
    edge_diag_dn = p[:, 0] / numpy.sqrt(3) > -p[:, 1] + num_mirror + 1.1
    p = p[~edge_diag_up & ~edge_diag_dn, :]
    return p


def y_splitter(num_mirror: int) -> NDArray[numpy.float64]:
    """
    Line defect waveguide y-splitter in a triangular lattice.

    Args:
        num_mirror: Mirror length (number of holes per side; total size is
            approximately `2 * n + 1` holes.

    Returns:
        `[[x0, y0], [x1, y1], ...]` for all the holes
    """
    p = triangular_lattice([4 * num_mirror + 1, 4 * num_mirror + 1])
    left_horiz = (p[:, 1] == 0) & (p[:, 0] <= 0)
    p = p[~left_horiz, :]

    # y = +-sqrt(3) * x
    right_diag_up = numpy.isclose(p[:, 1],  p[:, 0] * numpy.sqrt(3)) & (p[:, 0] >= 0)
    right_diag_dn = numpy.isclose(p[:, 1], -p[:, 0] * numpy.sqrt(3)) & (p[:, 0] >= 0)
    p = p[~right_diag_up & ~right_diag_dn, :]

    edge_left = p[:, 0] < -num_mirror
    p = p[~edge_left, :]

    edge_diag_up = p[:, 0] / numpy.sqrt(3) >  p[:, 1] + num_mirror + 1.1
    edge_diag_dn = p[:, 0] / numpy.sqrt(3) > -p[:, 1] + num_mirror + 1.1
    p = p[~edge_diag_up & ~edge_diag_dn, :]
    return p


def ln_defect(
        mirror_dims: Sequence[int],
        defect_length: int,
        ) -> NDArray[numpy.float64]:
    """
    N-hole defect in a triangular lattice.

    Args:
        mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is 2 * n + 1 in each direction.
        defect_length: Length of defect. Should be an odd number.

    Returns:
        `[[x0, y0], [x1, y1], ...]` for all the holes
    """
    if defect_length % 2 != 1:
        raise Exception('defect_length must be odd!')
    p = triangular_lattice([2 * d + 1 for d in mirror_dims])
    half_length = numpy.floor(defect_length / 2)
    hole_nums = numpy.arange(-half_length, half_length + 1)
    holes_to_keep = numpy.in1d(p[:, 0], hole_nums, invert=True)
    return p[numpy.logical_or(holes_to_keep, p[:, 1] != 0), ]


def ln_shift_defect(
        mirror_dims: Sequence[int],
        defect_length: int,
        shifts_a: ArrayLike = (0.15, 0, 0.075),
        shifts_r: ArrayLike = (1, 1, 1),
        ) -> NDArray[numpy.float64]:
    """
    N-hole defect with shifted holes (intended to give the mode a gaussian profile
     in real- and k-space so as to improve both Q and confinement). Holes along the
     defect line are shifted and altered according to the shifts_* parameters.

    Args:
        mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is `2 * n + 1` in each direction.
        defect_length: Length of defect. Should be an odd number.
        shifts_a: Percentage of a to shift (1st, 2nd, 3rd,...) holes along the defect line
        shifts_r: Factor to multiply the radius by. Should match length of shifts_a

    Returns:
        `[[x0, y0, r0], [x1, y1, r1], ...]` for all the holes
    """
    xy = ln_defect(mirror_dims, defect_length)

    # Add column for radius
    xyr = numpy.hstack((xy, numpy.ones((xy.shape[0], 1))))

    # Shift holes
    # Expand shifts as necessary
    tmp_a = numpy.array(shifts_a)
    tmp_r = numpy.array(shifts_r)
    n_shifted = max(tmp_a.size, tmp_r.size)

    shifts_a = numpy.ones(n_shifted)
    shifts_r = numpy.ones(n_shifted)
    shifts_a[:len(tmp_a)] = tmp_a
    shifts_r[:len(tmp_r)] = tmp_r

    x_removed = numpy.floor(defect_length / 2)

    for ind in range(n_shifted):
        for sign in (-1, 1):
            x_val = sign * (x_removed + ind + 1)
            which = numpy.logical_and(xyr[:, 0] == x_val, xyr[:, 1] == 0)
            xyr[which, ] = (x_val + numpy.sign(x_val) * shifts_a[ind], 0, shifts_r[ind])

    return xyr


def r6_defect(mirror_dims: Sequence[int]) -> NDArray[numpy.float64]:
    """
    R6 defect in a triangular lattice.

    Args:
        mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is 2 * n + 1 in each direction.

    Returns:
        `[[x0, y0], [x1, y1], ...]` specifying hole centers.
    """
    xy = triangular_lattice([2 * d + 1 for d in mirror_dims])

    rem_holes_plus = numpy.array([[1, 0],
                                  [0.5, +numpy.sqrt(3)/2],
                                  [0.5, -numpy.sqrt(3)/2]])
    rem_holes = numpy.vstack((rem_holes_plus, -rem_holes_plus))

    for rem_xy in rem_holes:
        xy = xy[(xy != rem_xy).any(axis=1), ]

    return xy


def l3_shift_perturbed_defect(
        mirror_dims: Sequence[int],
        perturbed_radius: float = 1.1,
        shifts_a: Sequence[float] = (),
        shifts_r: Sequence[float] = ()
        ) -> NDArray[numpy.float64]:
    """
    3-hole defect with perturbed hole sizes intended to form an upwards-directed
     beam. Can also include shifted holes along the defect line, intended
     to give the mode a more gaussian profile to improve Q.

    Args:
        mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is 2 * n + 1 in each direction.
        perturbed_radius: Amount to perturb the radius of the holes used for beam-forming
        shifts_a: Percentage of a to shift (1st, 2nd, 3rd,...) holes along the defect line
        shifts_r: Factor to multiply the radius by. Should match length of shifts_a

    Returns:
        `[[x0, y0, r0], [x1, y1, r1], ...]` for all the holes
    """
    xyr = ln_shift_defect(mirror_dims, 3, shifts_a, shifts_r)

    abs_x, abs_y = (numpy.fabs(xyr[:, i]) for i in (0, 1))

    # Sorted unique xs and ys
    # Ignore row y=0 because it might have shifted holes
    xs = numpy.unique(abs_x[abs_x != 0])
    ys = numpy.unique(abs_y)

    # which holes should be perturbed? (xs[[3, 7]], ys[1]) and (xs[[2, 6]], ys[2])
    perturbed_holes = ((xs[a], ys[b]) for a, b in ((3, 1), (7, 1), (2, 2), (6, 2)))
    for row in xyr:
        if numpy.fabs(row) in perturbed_holes:
            row[2] = perturbed_radius
    return xyr
