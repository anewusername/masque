"""
2D bin-packing
"""
from typing import Sequence, Callable, Mapping

import numpy
from numpy.typing import NDArray, ArrayLike

from ..error import MasqueError
from ..pattern import Pattern


def maxrects_bssf(
        rects: ArrayLike,
        containers: ArrayLike,
        presort: bool = True,
        allow_rejects: bool = True,
        ) -> tuple[NDArray[numpy.float64], set[int]]:
    """
    Pack rectangles `rects` into regions `containers` using the "maximal rectangles best short side fit"
    algorithm (maxrects_bssf) from "A thousand ways to pack the bin", Jukka Jylanki, 2010.

    This algorithm gives the best results, but is asymptotically slower than `guillotine_bssf_sas`.

    Args:
        rects: Nx2 array of rectangle sizes `[[x_size0, y_size0], ...]`.
        containers: Mx4 array of regions into which `rects` will be placed, specified using their
            corner coordinates ` [[x_min0, y_min0, x_max0, y_max0], ...]`.
        presort: If `True` (default), largest-shortest-side rectangles will be placed
            first. Otherwise, they will be placed in the order provided.
        allow_rejects: If `False`, `MasqueError` will be raised if any rectangle cannot be placed.

    Returns:
        `[[x_min0, y_min0], ...]` placement locations for `rects`, with the same ordering.
        The second argument is a set of indicies of `rects` entries which were rejected; their
        corresponding placement locations should be ignored.

    Raises:
        MasqueError if `allow_rejects` is `True` but some `rects` could not be placed.
    """
    regions = numpy.array(containers, copy=False, dtype=float)
    rect_sizes = numpy.array(rects, copy=False, dtype=float)
    rect_locs = numpy.zeros_like(rect_sizes)
    rejected_inds = set()

    if presort:
        rotated_sizes = numpy.sort(rect_sizes, axis=1)  # shortest side first
        rect_order = numpy.lexsort(rotated_sizes.T)[::-1]  # Descending shortest side
        rect_sizes = rect_sizes[rect_order]

    for rect_ind, rect_size in enumerate(rect_sizes):
        ''' Remove degenerate regions '''
        # First remove duplicate regions (but keep one; code below would drop both)
        regions = numpy.unique(regions, axis=0)

        # Now remove regions enclosed in another
        min_more = (regions[None, :, :2] >= regions[:, None, :2]).all(axis=2)   # first axis > second axis
        max_less = (regions[None, :, 2:] <= regions[:, None, 2:]).all(axis=2)   # first axis < second axis
        max_less &= ~numpy.eye(regions.shape[0], dtype=bool)    # exclude self
        degenerate = (min_more & max_less).any(axis=0)
        regions = regions[~degenerate]

        ''' Place the rect '''
        # Best short-side fit (bssf) to pick a region
        bssf_scores = ((regions[:, 2:] - regions[:, :2]) - rect_size).min(axis=1).astype(float)
        bssf_scores[bssf_scores < 0] = numpy.inf        # doesn't fit!
        rr = bssf_scores.argmin()
        if numpy.isinf(bssf_scores[rr]):
            if allow_rejects:
                rejected_inds.add(rect_ind)
                continue
            else:
                raise MasqueError(f'Failed to find a suitable location for rectangle {rect_ind}')

        # Read out location
        loc = regions[rr, :2]
        rect_locs[rect_ind] = loc

        ''' Shatter regions '''
        # Which regions does this rectangle intersect?
        min_over = regions[:, :2] >= loc + rect_size
        max_undr = regions[:, 2:] <= loc
        intersects = ~(min_over | max_undr).any(axis=1)

        # Which sides is there excess on?
        region_past_botleft = intersects[:, None] & (regions[:, :2] < loc)
        region_past_topright = intersects[:, None] & (regions[:, 2:] > loc + rect_size)

        # Create new regions
        r_lft = regions[region_past_botleft[:, 0]].copy()
        r_bot = regions[region_past_botleft[:, 1]].copy()
        r_rgt = regions[region_past_topright[:, 0]].copy()
        r_top = regions[region_past_topright[:, 1]].copy()

        r_lft[:, 2] = loc[0]
        r_bot[:, 3] = loc[1]
        r_rgt[:, 0] = loc[0] + rect_size[0]
        r_top[:, 1] = loc[1] + rect_size[1]

        regions = numpy.vstack((regions[~intersects], r_lft, r_bot, r_rgt, r_top))

    if presort:
        unsort_order = rect_order.argsort()
        rect_locs = rect_locs[unsort_order]
        rejected_inds = set(unsort_order[list(rejected_inds)])

    return rect_locs, rejected_inds


def guillotine_bssf_sas(
        rects: ArrayLike,
        containers: ArrayLike,
        presort: bool = True,
        allow_rejects: bool = True,
        ) -> tuple[NDArray[numpy.float64], set[int]]:
    """
    Pack rectangles `rects` into regions `containers` using the "guillotine best short side fit with
    shorter axis split rule" algorithm (guillotine-BSSF-SAS) from "A thousand ways to pack the bin",
    Jukka Jylanki, 2010.

    This algorithm gives the worse results than `maxrects_bssf`, but is asymptotically faster.

    # TODO consider adding rectangle-merge?
    # TODO guillotine could use some additional testing

    Args:
        rects: Nx2 array of rectangle sizes `[[x_size0, y_size0], ...]`.
        containers: Mx4 array of regions into which `rects` will be placed, specified using their
            corner coordinates ` [[x_min0, y_min0, x_max0, y_max0], ...]`.
        presort: If `True` (default), largest-shortest-side rectangles will be placed
            first. Otherwise, they will be placed in the order provided.
        allow_rejects: If `False`, `MasqueError` will be raised if any rectangle cannot be placed.

    Returns:
        `[[x_min0, y_min0], ...]` placement locations for `rects`, with the same ordering.
        The second argument is a set of indicies of `rects` entries which were rejected; their
        corresponding placement locations should be ignored.

    Raises:
        MasqueError if `allow_rejects` is `True` but some `rects` could not be placed.
    """
    regions = numpy.array(containers, copy=False, dtype=float)
    rect_sizes = numpy.array(rects, copy=False, dtype=float)
    rect_locs = numpy.zeros_like(rect_sizes)
    rejected_inds = set()

    if presort:
        rotated_sizes = numpy.sort(rect_sizes, axis=1)  # shortest side first
        rect_order = numpy.lexsort(rotated_sizes.T)[::-1]  # Descending shortest side
        rect_sizes = rect_sizes[rect_order]

    for rect_ind, rect_size in enumerate(rect_sizes):
        ''' Place the rect '''
        # Best short-side fit (bssf) to pick a region
        bssf_scores = ((regions[:, 2:] - regions[:, :2]) - rect_size).min(axis=1).astype(float)
        bssf_scores[bssf_scores < 0] = numpy.inf        # doesn't fit!
        rr = bssf_scores.argmin()
        if numpy.isinf(bssf_scores[rr]):
            if allow_rejects:
                rejected_inds.add(rect_ind)
                continue
            else:
                raise MasqueError(f'Failed to find a suitable location for rectangle {rect_ind}')

        # Read out location
        loc = regions[rr, :2]
        rect_locs[rect_ind] = loc

        region_size = regions[rr, 2:] - loc
        split_horiz = region_size[0] < region_size[1]

        new_region0 = regions[rr].copy()
        new_region1 = new_region0.copy()
        split_vertex = loc + rect_size
        if split_horiz:
            new_region0[2] = split_vertex[0]
            new_region0[1] = split_vertex[1]
            new_region1[0] = split_vertex[0]
        else:
            new_region0[3] = split_vertex[1]
            new_region0[0] = split_vertex[0]
            new_region1[1] = split_vertex[1]

        regions = numpy.vstack((regions[:rr], regions[rr + 1:],
                                new_region0, new_region1))

    if presort:
        unsort_order = rect_order.argsort()
        rect_locs = rect_locs[unsort_order]
        rejected_inds = set(unsort_order[list(rejected_inds)])

    return rect_locs, rejected_inds


def pack_patterns(
        library: Mapping[str, Pattern],
        patterns: Sequence[str],
        containers: ArrayLike,
        spacing: tuple[float, float],
        presort: bool = True,
        allow_rejects: bool = True,
        packer: Callable = maxrects_bssf,
        ) -> tuple[Pattern, list[str]]:
    """
    Pick placement locations for `patterns` inside the regions specified by `containers`.
    No rotations are performed.

    Args:
        library: Library from which `Pattern` objects will be drawn.
        patterns: Sequence of pattern names which are to be placed.
        containers: Mx4 array of regions into which `patterns` will be placed, specified using their
            corner coordinates ` [[x_min0, y_min0, x_max0, y_max0], ...]`.
        spacing: (x, y) spacing between adjacent patterns. Patterns are effectively expanded outwards
            by `spacing / 2` prior to placement, so this also affects pattern position relative to
            container edges.
        presort: If `True` (default), largest-shortest-side rectangles will be placed
            first. Otherwise, they will be placed in the order provided.
        allow_rejects: If `False`, `MasqueError` will be raised if any rectangle cannot be placed.
        packer: Bin-packing method; see the other functions in this module (namely `maxrects_bssf`
            and `guillotine_bssf_sas`).

    Returns:
        A `Pattern` containing one `Ref` for each entry in `patterns`.
        A list of "rejected" pattern names, for which a valid placement location could not be found.

    Raises:
        MasqueError if `allow_rejects` is `True` but some `rects` could not be placed.
    """

    half_spacing = numpy.array(spacing, copy=False, dtype=float) / 2

    bounds = [library[pp].get_bounds() for pp in patterns]
    sizes = [bb[1] - bb[0] + spacing if bb is not None else spacing for bb in bounds]
    offsets = [half_spacing - bb[0] if bb is not None else (0, 0) for bb in bounds]

    locations, reject_inds = packer(sizes, containers, presort=presort, allow_rejects=allow_rejects)

    pat = Pattern()
    for pp, oo, loc in zip(patterns, offsets, locations):
        pat.ref(pp, offset=oo + loc)

    rejects = [patterns[ii] for ii in reject_inds]
    return pat, rejects
