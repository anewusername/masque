"""
2D bin-packing
"""
from typing import Tuple, List, Set, Sequence, Callable

import numpy
from numpy.typing import NDArray, ArrayLike

from ..error import MasqueError
from ..pattern import Pattern
from ..subpattern import SubPattern


def pack_patterns(patterns: Sequence[Pattern],
                  regions: numpy.ndarray,
                  spacing: Tuple[float, float],
                  presort: bool = True,
                  allow_rejects: bool = True,
                  packer: Callable = maxrects_bssf,
                  ) -> Tuple[Pattern, List[Pattern]]:
    half_spacing = numpy.array(spacing) / 2

    bounds = [pp.get_bounds() for pp in patterns]
    sizes = [bb[1] - bb[0] + spacing if bb is not None else spacing for bb in bounds]
    offsets = [half_spacing - bb[0] if bb is not None else (0, 0) for bb in bounds]

    locations, reject_inds = packer(sizes, regions, presort=presort, allow_rejects=allow_rejects)

    pat = Pattern()
    pat.subpatterns = [SubPattern(pp, offset=oo + loc)
                       for pp, oo, loc in zip(patterns, offsets, locations)]

    rejects = [patterns[ii] for ii in reject_inds]
    return pat, rejects


def maxrects_bssf(
        rects: ArrayLike,
        containers: ArrayLike,
        presort: bool = True,
        allow_rejects: bool = True,
        ) -> Tuple[NDArray[numpy.float64], Set[int]]:
    """
    sizes should be Nx2
    regions should be Mx4 (xmin, ymin, xmax, ymax)
    """
    regions = numpy.array(containers, copy=False, dtype=float)
    rect_sizes = numpy.array(rects, copy=False, dtype=float)
    rect_locs = numpy.zeros_like(rect_sizes)
    rejected_inds = set()

    if presort:
        rotated_sizes = numpy.sort(rect_sizes, axis=0)  # shortest side first
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
    return rect_locs, rejected_inds


def guillotine_bssf_sas(rect_sizes: numpy.ndarray,
                        regions: numpy.ndarray,
                        presort: bool = True,
                        allow_rejects: bool = True,
                        ) -> Tuple[numpy.ndarray, Set[int]]:
    """
    sizes should be Nx2
    regions should be Mx4 (xmin, ymin, xmax, ymax)
    #TODO: test me!
    # TODO add rectangle-merge?
    """
    rect_sizes = numpy.array(rect_sizes)
    rect_locs = numpy.zeros_like(rect_sizes)
    rejected_inds = set()

    if presort:
        rotated_sizes = numpy.sort(rect_sizes, axis=0)  # shortest side first
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
        split_vert = loc + rect_size
        if split_horiz:
            new_region0[2] = split_vert[0]
            new_region0[1] = split_vert[1]
            new_region1[0] = split_vert[0]
        else:
            new_region0[3] = split_vert[1]
            new_region0[0] = split_vert[0]
            new_region1[1] = split_vert[1]

        regions = numpy.vstack((regions[:rr], regions[rr + 1:],
                                new_region0, new_region1))

    return rect_locs, rejected_inds
