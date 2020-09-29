"""
Helper functions for file reading and writing
"""
from typing import Set, Tuple, List
import re
import copy
import gzip
import pathlib

from .. import Pattern, PatternError
from ..shapes import Polygon, Path


def mangle_name(pattern: Pattern, dose_multiplier: float=1.0) -> str:
    """
    Create a name using `pattern.name`, `id(pattern)`, and the dose multiplier.

    Args:
        pattern: Pattern whose name we want to mangle.
        dose_multiplier: Dose multiplier to mangle with.

    Returns:
        Mangled name.
    """
    expression = re.compile('[^A-Za-z0-9_\?\$]')
    full_name = '{}_{}_{}'.format(pattern.name, dose_multiplier, id(pattern))
    sanitized_name = expression.sub('_', full_name)
    return sanitized_name


def clean_pattern_vertices(pat: Pattern) -> Pattern:
    """
    Given a pattern, remove any redundant vertices in its polygons and paths.
    The cleaning process completely removes any polygons with zero area or <3 vertices.

    Args:
        pat: Pattern to clean

    Returns:
        pat
    """
    remove_inds = []
    for ii, shape in enumerate(pat.shapes):
        if not isinstance(shape, (Polygon, Path)):
            continue
        try:
            shape.clean_vertices()
        except PatternError:
            remove_inds.append(ii)
    for ii in sorted(remove_inds, reverse=True):
        del pat.shapes[ii]
    return pat


def make_dose_table(patterns: List[Pattern], dose_multiplier: float=1.0) -> Set[Tuple[int, float]]:
    """
    Create a set containing `(id(pat), written_dose)` for each pattern (including subpatterns)

    Args:
        pattern: Source Patterns.
        dose_multiplier: Multiplier for all written_dose entries.

    Returns:
        `{(id(subpat.pattern), written_dose), ...}`
    """
    dose_table = {(id(pattern), dose_multiplier) for pattern in patterns}
    for pattern in patterns:
        for subpat in pattern.subpatterns:
            if subpat.pattern is None:
                continue
            subpat_dose_entry = (id(subpat.pattern), subpat.dose * dose_multiplier)
            if subpat_dose_entry not in dose_table:
                subpat_dose_table = make_dose_table([subpat.pattern], subpat.dose * dose_multiplier)
                dose_table = dose_table.union(subpat_dose_table)
    return dose_table


def dtype2dose(pattern: Pattern) -> Pattern:
    """
    For each shape in the pattern, if the layer is a tuple, set the
      layer to the tuple's first element and set the dose to the
      tuple's second element.

    Generally intended for use with `Pattern.apply()`.

    Args:
        pattern: Pattern to modify

    Returns:
        pattern
    """
    for shape in pattern.shapes:
        if isinstance(shape.layer, tuple):
            shape.dose = shape.layer[1]
            shape.layer = shape.layer[0]
    return pattern


def dose2dtype(patterns: List[Pattern],
               ) -> Tuple[List[Pattern], List[float]]:
    """
     For each shape in each pattern, set shape.layer to the tuple
     (base_layer, datatype), where:
        layer is chosen to be equal to the original shape.layer if it is an int,
            or shape.layer[0] if it is a tuple. `str` layers raise a PatterError.
        datatype is chosen arbitrarily, based on calcualted dose for each shape.
            Shapes with equal calcualted dose will have the same datatype.
            A list of doses is retured, providing a mapping between datatype
            (list index) and dose (list entry).

    Note that this function modifies the input Pattern(s).

    Args:
        patterns: A `Pattern` or list of patterns to write to file. Modified by this function.

    Returns:
        (patterns, dose_list)
            patterns: modified input patterns
            dose_list: A list of doses, providing a mapping between datatype (int, list index)
                       and dose (float, list entry).
    """
    # Get a dict of id(pattern) -> pattern
    patterns_by_id = {id(pattern): pattern for pattern in patterns}
    for pattern in patterns:
        for i, p in pattern.referenced_patterns_by_id().items():
            patterns_by_id[i] = p

    # Get a table of (id(pat), written_dose) for each pattern and subpattern
    sd_table = make_dose_table(patterns)

    # Figure out all the unique doses necessary to write this pattern
    #  This means going through each row in sd_table and adding the dose values needed to write
    #  that subpattern at that dose level
    dose_vals = set()
    for pat_id, pat_dose in sd_table:
        pat = patterns_by_id[pat_id]
        for shape in pat.shapes:
            dose_vals.add(shape.dose * pat_dose)

    if len(dose_vals) > 256:
        raise PatternError('Too many dose values: {}, maximum 256 when using dtypes.'.format(len(dose_vals)))

    dose_vals_list = list(dose_vals)

    # Create a new pattern for each non-1-dose entry in the dose table
    #   and update the shapes to reflect their new dose
    new_pats = {} # (id, dose) -> new_pattern mapping
    for pat_id, pat_dose in sd_table:
        if pat_dose == 1:
            new_pats[(pat_id, pat_dose)] = patterns_by_id[pat_id]
            continue

        old_pat = patterns_by_id[pat_id]
        pat = old_pat.copy() # keep old subpatterns
        pat.shapes = copy.deepcopy(old_pat.shapes)
        pat.labels = copy.deepcopy(old_pat.labels)

        encoded_name = mangle_name(pat, pat_dose)
        if len(encoded_name) == 0:
            raise PatternError('Zero-length name after mangle+encode, originally "{}"'.format(pat.name))
        pat.name = encoded_name

        for shape in pat.shapes:
            data_type = dose_vals_list.index(shape.dose * pat_dose)
            if isinstance(shape.layer, int):
                shape.layer = (shape.layer, data_type)
            elif isinstance(shape.layer, tuple):
                shape.layer = (shape.layer[0], data_type)
            else:
                raise PatternError(f'Invalid layer for gdsii: {shape.layer}')

        new_pats[(pat_id, pat_dose)] = pat

    # Go back through all the dose-specific patterns and fix up their subpattern entries
    for (pat_id, pat_dose), pat in new_pats.items():
        for subpat in pat.subpatterns:
            dose_mult = subpat.dose * pat_dose
            subpat.pattern = new_pats[(id(subpat.pattern), dose_mult)]

    return patterns, dose_vals_list


def is_gzipped(path: pathlib.Path) -> bool:
    with open(path, 'rb') as stream:
        magic_bytes = stream.read(2)
        return magic_bytes == b'\x1f\x8b'
