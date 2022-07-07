"""
Library class for managing unique name->pattern mappings and
 deferred loading or creation.
"""
from typing import Dict, Callable, TypeVar, TYPE_CHECKING
from typing import Any, Tuple, Union, Iterator
import logging
from pprint import pformat
import copy

from ..error import LibraryError

if TYPE_CHECKING:
    from ..pattern import Pattern


logger = logging.getLogger(__name__)


L = TypeVar('L', bound='Library')


class Library:
    """
    This class is usually used to create a library of Patterns by mapping names to
     functions which generate or load the relevant `Pattern` object as-needed.

    Generated/loaded patterns can have "symbolic" references, where a SubPattern
     object `sp` has a `None`-valued `sp.pattern` attribute, in which case the
     Library expects `sp.identifier[0]` to contain a string which specifies the
     referenced pattern's name.

    The cache can be disabled by setting the `enable_cache` attribute to `False`.
    """
    generators: Dict[str, Callable[[], 'Pattern']]
    cache: Dict[str, 'Pattern']
    enable_cache: bool = True

    def __init__(self) -> None:
        self.generators = {}
        self.cache = {}

    def __setitem__(self, key: str, value: Callable[[], 'Pattern']) -> None:
        self.generators[key] = value
        if key in self.cache:
            logger.warning(f'Replaced library item "{key}" & existing cache entry.'
                           ' Previously-generated Pattern will *not* be updated!')
            del self.cache[key]

    def __delitem__(self, key: str) -> None:
        del self.generators[key]

        if key in self.cache:
            logger.warning(f'Deleting library item "{key}" & existing cache entry.'
                           ' Previously-generated Pattern may remain in the wild!')
            del self.cache[key]

    def __getitem__(self, key: str) -> 'Pattern':
        if self.enable_cache and key in self.cache:
            logger.debug(f'found {key} in cache')
            return self.cache[key]

        logger.debug(f'loading {key}')
        pat = self.generators[key]()
        self.resolve_subpatterns(pat)
        self.cache[key] = pat
        return pat

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.generators

    def resolve_subpatterns(self, pat: 'Pattern', tag: str) -> 'Pattern':
        logger.debug(f'Resolving subpatterns in {pat.name}')
        for sp in pat.subpatterns:
            if sp.pattern is not None:
                continue

            key = sp.identifier[0]
            if key in self.generators:
                sp.pattern = self[key]
                continue

            raise LibraryError(f'Broken reference to {key} (tag {tag})')
        return pat

    def keys(self) -> Iterator[str]:
        return iter(self.generators.keys())

    def values(self) -> Iterator['Pattern']:
        return iter(self[key] for key in self.keys())

    def items(self) -> Iterator[Tuple[str, 'Pattern']]:
        return iter((key, self[key]) for key in self.keys())

    def __repr__(self) -> str:
        return '<Library with keys ' + repr(list(self.generators.keys())) + '>'

    def precache(self: L) -> L:
        """
        Force all patterns into the cache

        Returns:
            self
        """
        for key in self.generators:
            _ = self[key]
        return self

    def add(
            self: L,
            other: L,
            use_ours: Callable[[str], bool] = lambda name: False,
            use_theirs: Callable[[str], bool] = lambda name: False,
            ) -> L:
        """
        Add keys from another library into this one.

        Args:
            other: The library to insert keys from
            use_ours: Decision function for name conflicts.
                Should return `True` if the value from `self` should be used.
            use_theirs: Decision function for name conflicts. Same format as `use_ours`.
                Should return `True` if the value from `other` should be used.
                `use_ours` takes priority over `use_theirs`.
        Returns:
            self
        """
        duplicates = set(self.keys()) & set(other.keys())
        keep_ours = set(name for name in duplicates if use_ours(name))
        keep_theirs = set(name for name in duplicates - keep_ours if use_theirs(name))
        conflicts = duplicates - keep_ours - keep_theirs

        if conflicts:
            raise LibraryError('Unresolved duplicate keys encountered in library merge: '
                               + pformat(conflicts))

        for key in set(other.keys()) - keep_ours:
            self[key] = other.generators[key]
            if key in other.cache:
                self.cache[key] = other.cache[key]
        return self

    def copy(self, preserve_cache: bool = False) -> 'Library':
        """
        Create a copy of this `Library`.

        A shallow copy is made of the contained dicts.
        Note that you should probably clear the cache (with `clear_cache()`) after copying.

        Returns:
            A copy of self
        """
        new = Library()
        new.generators.update(self.generators)
        new.cache.update(self.cache)
        return new

    def clear_cache(self: L) -> L:
        """
        Clear the cache of this library.
        This is usually used before modifying or deleting cells, e.g. when merging
          with another library.

        Returns:
            self
        """
        self.cache = {}
        return self

    def __deepcopy__(self: L, memo: Optional[Dict] = None) -> L:
        raise LibraryError('Library cannot be deepcopied -- python copy.deepcopy() does not copy closures!')

