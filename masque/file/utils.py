"""
Helper functions for file reading and writing
"""
from typing import IO, Iterator
import re
import pathlib
import logging
import tempfile
import shutil
from contextlib import contextmanager

from .. import Pattern, PatternError
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
    for shapes in pat.shapes.values():
        remove_inds = []
        for ii, shape in enumerate(shapes):
            if not isinstance(shape, (Polygon, Path)):
                continue
            try:
                shape.clean_vertices()
            except PatternError:
                remove_inds.append(ii)
        for ii in sorted(remove_inds, reverse=True):
            del shapes[ii]
    return pat


def is_gzipped(path: pathlib.Path) -> bool:
    with open(path, 'rb') as stream:
        magic_bytes = stream.read(2)
        return magic_bytes == b'\x1f\x8b'


@contextmanager
def tmpfile(path: str | pathlib.Path) -> Iterator[IO[bytes]]:
    """
    Context manager which allows you to write to a temporary file,
    and move that file into its final location only after the write
    has finished.
    """
    path = pathlib.Path(path)
    suffixes = ''.join(path.suffixes)
    with tempfile.NamedTemporaryFile(suffix=suffixes, delete=False) as tmp_stream:
        yield tmp_stream

    try:
        shutil.move(tmp_stream.name, path)
    finally:
        pathlib.Path(tmp_stream.name).unlink(missing_ok=True)
