"""
Helper functions for file reading and writing
"""
from typing import Set, Tuple, List, Iterable, Mapping
import re
import copy
import pathlib
import logging

from .. import Pattern, PatternError
from ..library import Library, WrapROLibrary
from ..shapes import Polygon, Path


logger = logging.getLogger(__name__)


def mangle_name(name: str, dose_multiplier: float = 1.0) -> str:
    """
    Create a new name using `name` and the `dose_multiplier`.

    Args:
        name: Name we want to mangle.
        dose_multiplier: Dose multiplier to mangle with.

    Returns:
        Mangled name.
    """
    if dose_multiplier == 1:
        full_name = name
    else:
        full_name = f'{name}_dm{dose_multiplier}'
    expression = re.compile(r'[^A-Za-z0-9_\?\$]')
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


def make_dose_table(
        top_names: Iterable[str],
        library: Mapping[str, Pattern],
        dose_multiplier: float = 1.0,
        ) -> Set[Tuple[str, float]]:
    """
    Create a set containing `(name, written_dose)` for each pattern (including subpatterns)

    Args:
        top_names: Names of all topcells
        pattern: Source Patterns.
        dose_multiplier: Multiplier for all written_dose entries.

    Returns:
        `{(name, written_dose), ...}`
    """
    dose_table = {(top_name, dose_multiplier) for top_name in top_names}
    for name, pattern in library.items():
        for subpat in pattern.subpatterns:
            if subpat.target is None:
                continue
            subpat_dose_entry = (subpat.target, subpat.dose * dose_multiplier)
            if subpat_dose_entry not in dose_table:
                subpat_dose_table = make_dose_table(subpat.target, library, subpat.dose * dose_multiplier)
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


def dose2dtype(
        library: Mapping[str, Pattern],
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
    logger.warning('TODO: dose2dtype() needs to be tested!')

    if not isinstance(library, Library):
        library = WrapROLibrary(library)

    # Get a table of (id(pat), written_dose) for each pattern and subpattern
    sd_table = make_dose_table(library.find_toplevel(), library)

    # Figure out all the unique doses necessary to write this pattern
    #  This means going through each row in sd_table and adding the dose values needed to write
    #  that subpattern at that dose level
    dose_vals = set()
    for name, pat_dose in sd_table:
        pat = library[name]
        for shape in pat.shapes:
            dose_vals.add(shape.dose * pat_dose)

    if len(dose_vals) > 256:
        raise PatternError('Too many dose values: {}, maximum 256 when using dtypes.'.format(len(dose_vals)))

    dose_vals_list = list(dose_vals)

    # Create a new pattern for each non-1-dose entry in the dose table
    #   and update the shapes to reflect their new dose
    new_names = {}  # {(old name, dose): new name} mapping
    new_lib = {}  # {new_name: new_pattern} mapping
    for name, pat_dose in sd_table:
        mangled_name = mangle_name(name, pat_dose)
        new_names[(name, pat_dose)] = mangled_name

        old_pat = library[name]

        if pat_dose == 1:
            new_lib[mangled_name] = old_pat
            continue

        pat = old_pat.deepcopy()

        if len(mangled_name) == 0:
            raise PatternError(f'Zero-length name after mangle, originally "{name}"')

        for shape in pat.shapes:
            data_type = dose_vals_list.index(shape.dose * pat_dose)
            if isinstance(shape.layer, int):
                shape.layer = (shape.layer, data_type)
            elif isinstance(shape.layer, tuple):
                shape.layer = (shape.layer[0], data_type)
            else:
                raise PatternError(f'Invalid layer for gdsii: {shape.layer}')

        new_lib[mangled_name] = pat

    return new_lib, dose_vals_list


def is_gzipped(path: pathlib.Path) -> bool:
    with open(path, 'rb') as stream:
        magic_bytes = stream.read(2)
        return magic_bytes == b'\x1f\x8b'
