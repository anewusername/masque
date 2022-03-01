"""
Library class for managing unique name->pattern mappings and
 deferred loading or creation.
"""
from typing import Dict, Callable, TypeVar, TYPE_CHECKING
from typing import Any, Tuple, Union, Iterator
import logging
from pprint import pformat
from dataclasses import dataclass
import copy

from ..error import LibraryError

if TYPE_CHECKING:
    from ..pattern import Pattern


logger = logging.getLogger(__name__)


@dataclass
class PatternGenerator:
    __slots__ = ('tag', 'gen')
    tag: str
    """ Unique identifier for the source """

    gen: Callable[[], 'Pattern']
    """ Function which generates a pattern when called """


L = TypeVar('L', bound='Library')


class Library:
    """
    This class is usually used to create a library of Patterns by mapping names to
     functions which generate or load the relevant `Pattern` object as-needed.

    Generated/loaded patterns can have "symbolic" references, where a SubPattern
     object `sp` has a `None`-valued `sp.pattern` attribute, in which case the
     Library expects `sp.identifier[0]` to contain a string which specifies the
     referenced pattern's name.

    Patterns can either be "primary" (default) or "secondary". Both get the
     same deferred-load behavior, but "secondary" patterns may have conflicting
     names and are not accessible through basic []-indexing. They are only used
     to fill symbolic references in cases where there is no "primary" pattern
     available, and only if both the referencing and referenced pattern-generators'
     `tag` values match (i.e., only if they came from the same source).

    Primary patterns can be turned into secondary patterns with the `demote`
     method, `promote` performs the reverse (secondary -> primary) operation.

    The `set_const` and `set_value` methods provide an easy way to transparently
      construct PatternGenerator objects and directly set create "secondary"
      patterns.

    The cache can be disabled by setting the `enable_cache` attribute to `False`.
    """
    primary: Dict[str, PatternGenerator]
    secondary: Dict[Tuple[str, str], PatternGenerator]
    cache: Dict[Union[str, Tuple[str, str]], 'Pattern']
    enable_cache: bool = True

    def __init__(self) -> None:
        self.primary = {}
        self.secondary = {}
        self.cache = {}

    def __setitem__(self, key: str, value: PatternGenerator) -> None:
        self.primary[key] = value
        if key in self.cache:
            logger.warning(f'Replaced library item "{key}" & existing cache entry.'
                           ' Previously-generated Pattern will *not* be updated!')
            del self.cache[key]

    def __delitem__(self, key: str) -> None:
        if isinstance(key, str):
            del self.primary[key]
        elif isinstance(key, tuple):
            del self.secondary[key]

        if key in self.cache:
            logger.warning(f'Deleting library item "{key}" & existing cache entry.'
                           ' Previously-generated Pattern may remain in the wild!')
            del self.cache[key]

    def __getitem__(self, key: str) -> 'Pattern':
        return self.get_primary(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.primary

    def get_primary(self, key: str) -> 'Pattern':
        if self.enable_cache and key in self.cache:
            logger.debug(f'found {key} in cache')
            return self.cache[key]

        logger.debug(f'loading {key}')
        pg = self.primary[key]
        pat = pg.gen()
        self.resolve_subpatterns(pat, pg.tag)
        self.cache[key] = pat
        return pat

    def get_secondary(self, key: str, tag: str) -> 'Pattern':
        logger.debug(f'get_secondary({key}, {tag})')
        key2 = (key, tag)
        if self.enable_cache and key2 in self.cache:
            return self.cache[key2]

        pg = self.secondary[key2]
        pat = pg.gen()
        self.resolve_subpatterns(pat, pg.tag)
        self.cache[key2] = pat
        return pat

    def set_secondary(self, key: str, tag: str, value: PatternGenerator) -> None:
        self.secondary[(key, tag)] = value
        if (key, tag) in self.cache:
            logger.warning(f'Replaced library item "{key}" & existing cache entry.'
                           ' Previously-generated Pattern will *not* be updated!')
            del self.cache[(key, tag)]

    def resolve_subpatterns(self, pat: 'Pattern', tag: str) -> 'Pattern':
        logger.debug(f'Resolving subpatterns in {pat.name}')
        for sp in pat.subpatterns:
            if sp.pattern is not None:
                continue

            key = sp.identifier[0]
            if key in self.primary:
                sp.pattern = self.get_primary(key)
                continue

            if (key, tag) in self.secondary:
                sp.pattern = self.get_secondary(key, tag)
                continue

            raise LibraryError(f'Broken reference to {key} (tag {tag})')
        return pat

    def keys(self) -> Iterator[str]:
        return iter(self.primary.keys())

    def values(self) -> Iterator['Pattern']:
        return iter(self[key] for key in self.keys())

    def items(self) -> Iterator[Tuple[str, 'Pattern']]:
        return iter((key, self[key]) for key in self.keys())

    def __repr__(self) -> str:
        return '<Library with keys ' + repr(list(self.primary.keys())) + '>'

    def set_const(
            self,
            key: str,
            tag: Any,
            const: 'Pattern',
            secondary: bool = False,
            ) -> None:
        """
        Convenience function to avoid having to manually wrap
         constant values into callables.

        Args:
            key: Lookup key, usually the cell/pattern name
            tag: Unique tag for the source, used to disambiguate secondary patterns
            const: Pattern object to return
            secondary: If True, this pattern is not accessible for normal lookup, and is
                        only used as a sub-component of other patterns if no non-secondary
                        equivalent is available.
        """
        pg = PatternGenerator(tag=tag, gen=lambda: const)
        if secondary:
            self.secondary[(key, tag)] = pg
        else:
            self.primary[key] = pg

    def set_value(
            self,
            key: str,
            tag: str,
            value: Callable[[], 'Pattern'],
            secondary: bool = False,
            ) -> None:
        """
        Convenience function to automatically build a PatternGenerator.

        Args:
            key: Lookup key, usually the cell/pattern name
            tag: Unique tag for the source, used to disambiguate secondary patterns
            value: Callable which takes no arguments and generates the `Pattern` object
            secondary: If True, this pattern is not accessible for normal lookup, and is
                        only used as a sub-component of other patterns if no non-secondary
                        equivalent is available.
        """
        pg = PatternGenerator(tag=tag, gen=value)
        if secondary:
            self.secondary[(key, tag)] = pg
        else:
            self.primary[key] = pg

    def precache(self: L) -> L:
        """
        Force all patterns into the cache

        Returns:
            self
        """
        for key in self.primary:
            _ = self.get_primary(key)
        for key2 in self.secondary:
            _ = self.get_secondary(*key2)
        return self

    def add(
            self: L,
            other: L,
            use_ours: Callable[[Union[str, Tuple[str, str]]], bool] = lambda name: False,
            use_theirs: Callable[[Union[str, Tuple[str, str]]], bool] = lambda name: False,
            ) -> L:
        """
        Add keys from another library into this one.

        Args:
            other: The library to insert keys from
            use_ours: Decision function for name conflicts.
                May be called with cell names and (name, tag) tuples for primary or
                secondary cells, respectively.
                Should return `True` if the value from `self` should be used.
            use_theirs: Decision function for name conflicts. Same format as `use_ours`.
                Should return `True` if the value from `other` should be used.
                `use_ours` takes priority over `use_theirs`.
        Returns:
            self
        """
        duplicates1 = set(self.primary.keys()) & set(other.primary.keys())
        duplicates2 = set(self.secondary.keys()) & set(other.secondary.keys())
        keep_ours1 = set(name for name in duplicates1 if use_ours(name))
        keep_ours2 = set(name for name in duplicates2 if use_ours(name))
        keep_theirs1 = set(name for name in duplicates1 - keep_ours1 if use_theirs(name))
        keep_theirs2 = set(name for name in duplicates2 - keep_ours2 if use_theirs(name))
        conflicts1 = duplicates1 - keep_ours1 - keep_theirs1
        conflicts2 = duplicates2 - keep_ours2 - keep_theirs2

        if conflicts1:
            raise LibraryError('Unresolved duplicate keys encountered in library merge: ' + pformat(conflicts1))

        if conflicts2:
            raise LibraryError('Unresolved duplicate secondary keys encountered in library merge: ' + pformat(conflicts2))

        for key1 in set(other.primary.keys()) - keep_ours1:
            self[key1] = other.primary[key1]
            if key1 in other.cache:
                self.cache[key1] = other.cache[key1]

        for key2 in set(other.secondary.keys()) - keep_ours2:
            self.set_secondary(*key2, other.secondary[key2])
            if key2 in other.cache:
                self.cache[key2] = other.cache[key2]

        return self

    def demote(self, key: str) -> None:
        """
        Turn a primary pattern into a secondary one.
        It will no longer be accessible through [] indexing and will only be used to
          when referenced by other patterns from the same source, and only if no primary
          pattern with the same name exists.

        Args:
            key: Lookup key, usually the cell/pattern name
        """
        pg = self.primary[key]
        key2 = (key, pg.tag)
        self.secondary[key2] = pg
        if key in self.cache:
            self.cache[key2] = self.cache[key]
        del self[key]

    def promote(self, key: str, tag: str) -> None:
        """
        Turn a secondary pattern into a primary one.
        It will become accessible through [] indexing and will be used to satisfy any
          reference to a pattern with its key, regardless of tag.

        Args:
            key: Lookup key, usually the cell/pattern name
            tag: Unique tag for identifying the pattern's source, used to disambiguate
                    secondary patterns
        """
        if key in self.primary:
            raise LibraryError(f'Promoting ({key}, {tag}), but {key} already exists in primary!')

        key2 = (key, tag)
        pg = self.secondary[key2]
        self.primary[key] = pg
        if key2 in self.cache:
            self.cache[key] = self.cache[key2]
        del self.secondary[key2]
        del self.cache[key2]

    def copy(self, preserve_cache: bool = False) -> 'Library':
        """
        Create a copy of this `Library`.

        A shallow copy is made of the contained dicts.
        Note that you should probably clear the cache (with `clear_cache()`) after copying.

        Returns:
            A copy of self
        """
        new = Library()
        new.primary.update(self.primary)
        new.secondary.update(self.secondary)
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


r"""
    #   Add a filter for names which aren't added

  - Registration:
    - scanned files (tag=filename, gen_fn[stream, {name: pos}])
    - generator functions (tag='fn?', gen_fn[params])
    - merge decision function (based on tag and cell name, can be "neither") ??? neither=keep both, load using same tag!
  - Load process:
    - file:
        - read single cell
        - check subpat identifiers, and load stuff recursively based on those. If not present, load from same file??
    - function:
        - generate cell
        - traverse and check if we should load any subcells from elsewhere. replace if so.
            * should fn generate subcells at all, or register those separately and have us control flow? maybe ask us and generate itself if not present?

  - Scan all GDS files, save name -> (file, position). Keep the streams handy.
  - Merge all names. This requires subcell merge because we don't know hierarchy.
      - possibly include a "neither" option during merge, to deal with subcells. Means: just use parent's file.
"""
