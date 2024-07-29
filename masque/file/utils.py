"""
Helper functions for file reading and writing
"""
from typing import IO
from collections.abc import Iterator, Mapping
import re
import pathlib
import logging
import tempfile
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pprint import pformat
from itertools import chain

from .. import Pattern, PatternError, Library, LibraryError
from ..shapes import Polygon, Path


logger = logging.getLogger(__name__)


def preflight(
        lib: Library,
        sort: bool = True,
        sort_elements: bool = False,
        allow_dangling_refs: bool | None = None,
        allow_named_layers: bool = True,
        prune_empty_patterns: bool = False,
        wrap_repeated_shapes: bool = False,
        ) -> Library:
    """
    Run a standard set of useful operations and checks, usually done immediately prior
    to writing to a file (or immediately after reading).

    Args:
        sort: Whether to sort the patterns based on their names, and optionaly sort the pattern contents.
            Default True. Useful for reproducible builds.
        sort_elements: Whether to sort the pattern contents. Requires sort=True to run.
        allow_dangling_refs: If `None` (default), warns about any refs to patterns that are not
            in the provided library. If `True`, no check is performed; if `False`, a `LibraryError`
            is raised instead.
        allow_named_layers: If `False`, raises a `PatternError` if any layer is referred to by
            a string instead of a number (or tuple).
        prune_empty_patterns: Runs `Library.prune_empty()`, recursively deleting any empty patterns.
        wrap_repeated_shapes: Runs `Library.wrap_repeated_shapes()`, turning repeated shapes into
            repeated refs containing non-repeated shapes.

    Returns:
        `lib` or an equivalent sorted library
    """
    if sort:
        lib = Library(dict(sorted(
            (nn, pp.sort(sort_elements=sort_elements)) for nn, pp in lib.items()
            )))

    if not allow_dangling_refs:
        refs = lib.referenced_patterns()
        dangling = refs - set(lib.keys())
        if dangling:
            msg = 'Dangling refs found: ' + pformat(dangling)
            if allow_dangling_refs is None:
                logger.warning(msg)
            else:
                raise LibraryError(msg)

    if not allow_named_layers:
        named_layers: Mapping[str, set] = defaultdict(set)
        for name, pat in lib.items():
            for layer in chain(pat.shapes.keys(), pat.labels.keys()):
                if isinstance(layer, str):
                    named_layers[name].add(layer)
        named_layers = dict(named_layers)
        if named_layers:
            raise PatternError('Non-numeric layers found:' + pformat(named_layers))

    if prune_empty_patterns:
        pruned = lib.prune_empty()
        if pruned:
            logger.info(f'Preflight pruned {len(pruned)} empty patterns')
            logger.debug('Pruned: ' + pformat(pruned))
        else:
            logger.debug('Preflight found no empty patterns')

    if wrap_repeated_shapes:
        lib.wrap_repeated_shapes()

    return lib


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
            if not isinstance(shape, Polygon | Path):
                continue
            try:
                shape.clean_vertices()
            except PatternError:
                remove_inds.append(ii)
        for ii in sorted(remove_inds, reverse=True):
            del shapes[ii]
    return pat


def is_gzipped(path: pathlib.Path) -> bool:
    with path.open('rb') as stream:
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
