"""
Helper functions for file reading and writing
"""
import re
from typing import Set, Tuple

from masque.pattern import Pattern


__author__ = 'Jan Petykiewicz'


def mangle_name(pattern: Pattern, dose_multiplier: float=1.0) -> str:
    """
    Create a name using pattern.name, id(pattern), and the dose multiplier.

    :param pattern: Pattern whose name we want to mangle.
    :param dose_multiplier: Dose multiplier to mangle with.
    :return: Mangled name.
    """
    expression = re.compile('[^A-Za-z0-9_\?\$]')
    sanitized_name = expression.sub('_', pattern.name)
    full_name = '{}_{}_{}'.format(sanitized_name, dose_multiplier, id(pattern))
    return full_name


def make_dose_table(pattern: Pattern, dose_multiplier: float=1.0) -> Set[Tuple[int, float]]:
    """
    Create a set containing (id(subpat.pattern), written_dose) for each subpattern

    :param pattern: Source Pattern.
    :param dose_multiplier: Multiplier for all written_dose entries.
    :return: {(id(subpat.pattern), written_dose), ...}
    """
    dose_table = {(id(pattern), dose_multiplier)}
    for subpat in pattern.subpatterns:
        subpat_dose_entry = (id(subpat.pattern), subpat.dose * dose_multiplier)
        if subpat_dose_entry not in dose_table:
            subpat_dose_table = make_dose_table(subpat.pattern, subpat.dose * dose_multiplier)
            dose_table = dose_table.union(subpat_dose_table)
    return dose_table
