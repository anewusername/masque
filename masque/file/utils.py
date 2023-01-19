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


def mangle_name(name: str) -> str:
    """
    Sanitize a name.

    Args:
        name: Name we want to mangle.

    Returns:
        Mangled name.
    """
    expression = re.compile(r'[^A-Za-z0-9_\?\$]')
    sanitized_name = expression.sub('_', name)
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


def is_gzipped(path: pathlib.Path) -> bool:
    with open(path, 'rb') as stream:
        magic_bytes = stream.read(2)
        return magic_bytes == b'\x1f\x8b'
