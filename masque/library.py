"""
Library class for managing unique name->pattern mappings and
 deferred loading or creation.
"""
from typing import List, Dict, Callable, TypeVar, Generic, Type, TYPE_CHECKING
from typing import Tuple, Union, Iterator, Mapping, MutableMapping, Set, Optional, Sequence
import logging
import base64
import struct
import re
from pprint import pformat
from collections import defaultdict
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import ArrayLike

from .error import LibraryError, PatternError
from .utils import rotation_matrix_2d, normalize_mirror
from .shapes import Shape, Polygon
from .label import Label

if TYPE_CHECKING:
    from .pattern import Pattern
    from .builder import Abstract


logger = logging.getLogger(__name__)


visitor_function_t = Callable[..., 'Pattern']
L = TypeVar('L', bound='Library')
ML = TypeVar('ML', bound='MutableLibrary')
LL = TypeVar('LL', bound='LazyLibrary')


class Library(Mapping[str, Pattern], metaclass=ABCMeta):
    # inherited abstract functions
    #def __getitem__(self, key: str) -> 'Pattern':
    #def __iter__(self) -> Iterator[str]:
    #def __len__(self) -> int:

    #__contains__, keys, items, values, get, __eq__, __ne__ supplied by Mapping

    def abstract(self, name: str) -> 'Abstract':
        return Abstract(name=name, ports=self[name].ports)

    def __repr__(self) -> str:
        return '<Library with keys ' + repr(list(self.keys())) + '>'

    def dangling_references(
            self,
            tops: Union[None, str, Sequence[str]] = None,
            ) -> Set[Optional[str]]:
        """
        Get the set of all pattern names not present in the library but referenced
        by `tops`, recursively traversing any refs.

        If `tops` are not given, all patterns in the library are checked.

        Args:
            tops: Name(s) of the pattern(s) to check.
                Default is all patterns in the library.
            skip: Memo, set patterns which have already been traversed.

        Returns:
            Set of all referenced pattern names
        """
        if tops is None:
            tops = tuple(self.keys())

        referenced = self.referenced_patterns(tops)
        return referenced - set(self.keys())

    def referenced_patterns(
            self,
            tops: Union[None, str, Sequence[str]] = None,
            skip: Optional[Set[Optional[str]]] = None,
            ) -> Set[Optional[str]]:
        """
        Get the set of all pattern names referenced by `tops`. Recursively traverses into any refs.

        If `tops` are not given, all patterns in the library are checked.

        Args:
            tops: Name(s) of the pattern(s) to check.
                Default is all patterns in the library.
            skip: Memo, set patterns which have already been traversed.

        Returns:
            Set of all referenced pattern names
        """
        if tops is None:
            tops = tuple(self.keys())

        if skip is None:
            skip = set([None])

        if isinstance(tops, str):
            tops = (tops,)

        # Get referenced patterns for all tops
        targets = set()
        for top in set(tops):
            targets |= self[top].referenced_patterns()

        # Perform recursive lookups, but only once for each name
        for target in targets - skip:
            assert target is not None
            if target in self:
                targets |= self.referenced_patterns(target, skip=skip)
            skip.add(target)

        return targets

    def subtree(
            self,
            tops: Union[str, Sequence[str]],
            ) -> 'Library':
        """
         Return a new `Library`, containing only the specified patterns and the patterns they
        reference (recursively).

        Args:
            tops: Name(s) of patterns to keep

        Returns:
            A `WrapROLibrary` containing only `tops` and the patterns they reference.
        """
        keep: Set[str] = self.referenced_patterns(tops) - set((None,))      # type: ignore

        filtered = {kk: vv for kk, vv in self.items() if kk in keep}
        new = WrapROLibrary(filtered)
        return new

    def polygonize(
            self: L,
            poly_num_points: Optional[int] = None,
            poly_max_arclen: Optional[float] = None,
            ) -> L:
        """
        Calls `.polygonize(...)` on each pattern in this library.
        Arguments are passed on to `shape.to_polygons(...)`.

        Args:
            poly_num_points: Number of points to use for each polygon. Can be overridden by
                `poly_max_arclen` if that results in more points. Optional, defaults to shapes'
                internal defaults.
            poly_max_arclen: Maximum arclength which can be approximated by a single line
             segment. Optional, defaults to shapes' internal defaults.

        Returns:
            self
        """
        for pat in self.values():
            pat.polygonize(poly_num_points, poly_max_arclen)
        return self

    def manhattanize(
            self: L,
            grid_x: ArrayLike,
            grid_y: ArrayLike,
            ) -> L:
        """
        Calls `.manhattanize(grid_x, grid_y)` on each pattern in this library.

        Args:
            grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
            grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.

        Returns:
            self
        """
        for pat in self.values():
            pat.manhattanize(grid_x, grid_y)
        return self

    def flatten(
            self,
            tops: Union[str, Sequence[str]],
            ) -> Dict[str, 'Pattern']:
        """
        Removes all refs and adds equivalent shapes.
        Also flattens all referenced patterns.

        Args:
            tops: The pattern(s) to flattern.

        Returns:
            {name: flat_pattern} mapping for all flattened patterns.
        """
        if isinstance(tops, str):
            tops = (tops,)

        flattened: Dict[str, Optional['Pattern']] = {}

        def flatten_single(name) -> None:
            flattened[name] = None
            pat = self[name].deepcopy()

            for ref in pat.refs:
                target = ref.target
                if target is None:
                    continue

                if target not in flattened:
                    flatten_single(target)
                if flattened[target] is None:
                    raise PatternError(f'Circular reference in {name} to {target}')

                p = ref.as_pattern(pattern=flattened[target])
                pat.append(p)

            pat.refs.clear()
            flattened[name] = pat

        for top in tops:
            flatten_single(top)

        assert None not in flattened.values()
        return flattened    # type: ignore

    def get_name(
            self,
            name: str = '__',
            sanitize: bool = True,
            max_length: int = 32,
            quiet: bool = False,
            ) -> str:
        """
        Find a unique name for the pattern.

        This function may be overridden in a subclass or monkey-patched to fit the caller's requirements.

        Args:
            name: Preferred name for the pattern. Default '__'.
            sanitize: Allows only alphanumeric charaters and _?$. Replaces invalid characters with underscores.
            max_length: Names longer than this will be truncated.
            quiet: If `True`, suppress log messages.

        Returns:
            Name, unique within this library.
        """
        if sanitize:
            # Remove invalid characters
            sanitized_name = re.compile(r'[^A-Za-z0-9_\?\$]').sub('_', name)
        else:
            sanitized_name = name

        ii = 0
        suffixed_name = sanitized_name
        while suffixed_name in self or suffixed_name == '':
            suffix = base64.b64encode(struct.pack('>Q', ii), b'$?').decode('ASCII')

            suffixed_name = sanitized_name + '$' + suffix[:-1].lstrip('A')
            ii += 1

        if len(suffixed_name) > max_length:
            if name == '':
                raise LibraryError(f'No valid pattern names remaining within the specified {max_length=}')

            cropped_name = self.get_name(sanitized_name[:-1], sanitize=sanitize, max_length=max_length, quiet=True)
        else:
            cropped_name = suffixed_name

        if not quiet:
            logger.info(f'Requested name "{name}" changed to "{cropped_name}"')

        return cropped_name

    def find_toplevel(self) -> List[str]:
        """
        Return the list of all patterns that are not referenced by any other pattern in the library.

        Returns:
            A list of pattern names in which no pattern is referenced by any other pattern.
        """
        names = set(self.keys())
        not_toplevel: Set[Optional[str]] = set()
        for name in names:
            not_toplevel |= set(sp.target for sp in self[name].refs)

        toplevel = list(names - not_toplevel)
        return toplevel

    def dfs(
            self: L,
            top: str,
            visit_before: Optional[visitor_function_t] = None,
            visit_after: Optional[visitor_function_t] = None,
            *,
            hierarchy: Tuple[str, ...] = (),
            transform: Union[ArrayLike, bool, None] = False,
            memo: Optional[Dict] = None,
            ) -> L:
        """
        Convenience function.
        Performs a depth-first traversal of a pattern and its referenced patterns.
        At each pattern in the tree, the following sequence is called:
            ```
            hierarchy += (top,)
            current_pattern = visit_before(current_pattern, **vist_args)
            for sp in current_pattern.refs]
                self.dfs(sp.target, visit_before, visit_after,
                         hierarchy, updated_transform, memo)
            current_pattern = visit_after(current_pattern, **visit_args)
            ```
          where `visit_args` are
            `hierarchy`:  (top_pattern, L1_pattern, L2_pattern, ..., parent_pattern, current_pattern)
                          tuple of all parent-and-higher pattern names
            `transform`:  numpy.ndarray containing cumulative
                          [x_offset, y_offset, rotation (rad), mirror_x (0 or 1)]
                          for the instance being visited
            `memo`:  Arbitrary dict (not altered except by `visit_before()` and `visit_after()`)

        Args:
            top: Name of the pattern to start at (root node of the tree).
            visit_before: Function to call before traversing refs.
                Should accept a `Pattern` and `**visit_args`, and return the (possibly modified)
                pattern. Default `None` (not called).
            visit_after: Function to call after traversing refs.
                Should accept a `Pattern` and `**visit_args`, and return the (possibly modified)
                pattern. Default `None` (not called).
            transform: Initial value for `visit_args['transform']`.
                Can be `False`, in which case the transform is not calculated.
                `True` or `None` is interpreted as `[0, 0, 0, 0]`.
            memo: Arbitrary dict for use by `visit_*()` functions. Default `None` (empty dict).
            hierarchy: Tuple of patterns specifying the hierarchy above the current pattern.
                Appended to the start of the generated `visit_args['hierarchy']`.
                Default is an empty tuple.

        Returns:
            self
        """
        if memo is None:
            memo = {}

        if transform is None or transform is True:
            transform = numpy.zeros(4)
        elif transform is not False:
            transform = numpy.array(transform)

        if top in hierarchy:
            raise LibraryError('.dfs() called on pattern with circular reference')

        hierarchy += (top,)

        pat = self[top]
        if visit_before is not None:
            pat = visit_before(pat, hierarchy=hierarchy, memo=memo, transform=transform)

        for ref in pat.refs:
            if transform is not False:
                sign = numpy.ones(2)
                if transform[3]:
                    sign[1] = -1
                xy = numpy.dot(rotation_matrix_2d(transform[2]), ref.offset * sign)
                mirror_x, angle = normalize_mirror(ref.mirrored)
                angle += ref.rotation
                sp_transform = transform + (xy[0], xy[1], angle, mirror_x)
                sp_transform[3] %= 2
            else:
                sp_transform = False

            if ref.target is None:
                continue

            self.dfs(
                top=ref.target,
                visit_before=visit_before,
                visit_after=visit_after,
                transform=sp_transform,
                memo=memo,
                hierarchy=hierarchy,
                )

        if visit_after is not None:
            pat = visit_after(pat, hierarchy=hierarchy, memo=memo, transform=transform)

        if self[top] is not pat:
            if isinstance(self, MutableLibrary):
                self._set(top, pat)
            else:
                raise LibraryError('visit_* functions returned a new `Pattern` object'
                                   ' but the library is immutable')

        return self


VVV = TypeVar('VVV')


class MutableLibrary(Generic[VVV], Library, metaclass=ABCMeta):
    # inherited abstract functions
    #def __getitem__(self, key: str) -> 'Pattern':
    #def __iter__(self) -> Iterator[str]:
    #def __len__(self) -> int:

    @abstractmethod
    def __setitem__(self, key: str, value: VVV) -> None:
        pass

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        pass

    @abstractmethod
    def _set(self, key: str, value: 'Pattern') -> None:
        pass

    @abstractmethod
    def _merge(self, other: Mapping[str, 'Pattern'], key: str) -> None:
        pass

    def add(
            self: ML,
            other: Mapping[str, 'Pattern'],
            use_ours: Callable[[str], bool] = lambda name: False,
            use_theirs: Callable[[str], bool] = lambda name: False,
            ) -> ML:
        """
        Add keys from another library into this one.

        Args:
            other: The library to insert keys from
            use_ours: Decision function for name conflicts, called with pattern name.
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
            raise LibraryError('Unresolved duplicate keys encountered in library merge: ' + pformat(conflicts))

        for key in set(other.keys()) - keep_ours:
            self._merge(other, key)

        return self

    def dedup(
            self: ML,
            norm_value: int = int(1e6),
            exclude_types: Tuple[Type] = (Polygon,),
            label2name: Optional[Callable[[Tuple], str]] = None,
            threshold: int = 2,
            ) -> ML:
        """
        Iterates through all `Pattern`s. Within each `Pattern`, it iterates
         over all shapes, calling `.normalized_form(norm_value)` on them to retrieve a scale-,
         offset-, and rotation-independent form. Each shape whose normalized form appears
         more than once is removed and re-added using `Ref` objects referencing a newly-created
         `Pattern` containing only the normalized form of the shape.

        Note:
            The default norm_value was chosen to give a reasonable precision when using
            integer values for coordinates.

        Args:
            norm_value: Passed to `shape.normalized_form(norm_value)`. Default `1e6` (see function
                note)
            exclude_types: Shape types passed in this argument are always left untouched, for
                speed or convenience. Default: `(shapes.Polygon,)`
            label2name: Given a label tuple as returned by `shape.normalized_form(...)`, pick
                a name for the generated pattern. Default `self.get_name('_shape')`.
            threshold: Only replace shapes with refs if there will be at least this many
                instances.

        Returns:
            self
        """
        # This currently simplifies globally (same shape in different patterns is
        # merged into the same ref target).

        from .pattern import Pattern

        if exclude_types is None:
            exclude_types = ()

        if label2name is None:
            def label2name(label):
                return self.get_name('_shape_')
            #label2name = lambda label: self.get_name('_shape')

        shape_counts: MutableMapping[Tuple, int] = defaultdict(int)
        shape_funcs = {}

        # ## First pass ##
        #  Using the label tuple from `.normalized_form()` as a key, check how many of each shape
        # are present and store the shape function for each one
        for pat in tuple(self.values()):
            for i, shape in enumerate(pat.shapes):
                if not any(isinstance(shape, t) for t in exclude_types):
                    label, _values, func = shape.normalized_form(norm_value)
                    shape_funcs[label] = func
                    shape_counts[label] += 1

        shape_pats = {}
        for label, count in shape_counts.items():
            if count < threshold:
                continue

            shape_func = shape_funcs[label]
            shape_pat = Pattern(shapes=[shape_func()])
            shape_pats[label] = shape_pat

        # ## Second pass ##
        for pat in tuple(self.values()):
            #  Store `[(index_in_shapes, values_from_normalized_form), ...]` for all shapes which
            # are to be replaced.
            # The `values` are `(offset, scale, rotation)`.

            shape_table: MutableMapping[Tuple, List] = defaultdict(list)
            for i, shape in enumerate(pat.shapes):
                if any(isinstance(shape, t) for t in exclude_types):
                    continue

                label, values, _func = shape.normalized_form(norm_value)

                if label not in shape_pats:
                    continue

                shape_table[label].append((i, values))

            #  For repeated shapes, create a `Pattern` holding a normalized shape object,
            # and add `pat.refs` entries for each occurrence in pat. Also, note down that
            # we should delete the `pat.shapes` entries for which we made `Ref`s.
            shapes_to_remove = []
            for label in shape_table:
                target = label2name(label)
                for i, values in shape_table[label]:
                    offset, scale, rotation, mirror_x = values
                    pat.ref(target=target, offset=offset, scale=scale,
                            rotation=rotation, mirrored=(mirror_x, False))
                    shapes_to_remove.append(i)

            # Remove any shapes for which we have created refs.
            for i in sorted(shapes_to_remove, reverse=True):
                del pat.shapes[i]

        for ll, pp in shape_pats.items():
            self._set(label2name(ll), pp)

        return self

    def wrap_repeated_shapes(
            self: ML,
            name_func: Optional[Callable[['Pattern', Union[Shape, Label]], str]] = None,
            ) -> ML:
        """
        Wraps all shapes and labels with a non-`None` `repetition` attribute
          into a `Ref`/`Pattern` combination, and applies the `repetition`
          to each `Ref` instead of its contained shape.

        Args:
            name_func: Function f(this_pattern, shape) which generates a name for the
                        wrapping pattern. Default is `self.get_name('_rep')`.

        Returns:
            self
        """
        from .pattern import Pattern

        if name_func is None:
            def name_func(_pat, _shape):
                return self.get_name('_rep_')
            #name_func = lambda _pat, _shape: self.get_name('_rep')

        for pat in tuple(self.values()):
            new_shapes = []
            for shape in pat.shapes:
                if shape.repetition is None:
                    new_shapes.append(shape)
                    continue

                name = name_func(pat, shape)
                self._set(name, Pattern(shapes=[shape]))
                pat.ref(name, repetition=shape.repetition)
                shape.repetition = None
            pat.shapes = new_shapes

            new_labels = []
            for label in pat.labels:
                if label.repetition is None:
                    new_labels.append(label)
                    continue
                name = name_func(pat, label)
                self._set(name, Pattern(labels=[label]))
                pat.ref(name, repetition=label.repetition)
                label.repetition = None
            pat.labels = new_labels

        return self

    def subtree(
            self: ML,
            tops: Union[str, Sequence[str]],
            ) -> ML:
        """
         Return a new `Library`, containing only the specified patterns and the patterns they
        reference (recursively).

        Args:
            tops: Name(s) of patterns to keep

        Returns:
            A `Library` containing only `tops` and the patterns they reference.
        """
        keep: Set[str] = self.referenced_patterns(tops) - set((None,))      # type: ignore

        new = type(self)()
        for key in keep:
            new._merge(self, key)
        return new


class WrapROLibrary(Library):
    mapping: Mapping[str, 'Pattern']

    def __init__(
            self,
            mapping: Mapping[str, 'Pattern'],
            ) -> None:
        self.mapping = mapping

    def __getitem__(self, key: str) -> 'Pattern':
        return self.mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __repr__(self) -> str:
        return f'<WrapROLibrary ({type(self.mapping)}) with keys ' + repr(list(self.keys())) + '>'


class WrapLibrary(MutableLibrary):
    mapping: MutableMapping[str, 'Pattern']

    def __init__(
            self,
            mapping: MutableMapping[str, 'Pattern'],
            ) -> None:
        self.mapping = mapping

    def __getitem__(self, key: str) -> 'Pattern':
        return self.mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __setitem__(self, key: str, value: 'Pattern') -> None:
        self.mapping[key] = value

    def __delitem__(self, key: str) -> None:
        del self.mapping[key]

    def _set(self, key: str, value: 'Pattern') -> None:
        self[key] = value

    def _merge(self, other: Mapping[str, 'Pattern'], key: str) -> None:
        self[key] = other[key]

    def __repr__(self) -> str:
        return f'<WrapLibrary ({type(self.mapping)}) with keys ' + repr(list(self.keys())) + '>'


class LazyLibrary(MutableLibrary):
    """
    This class is usually used to create a library of Patterns by mapping names to
     functions which generate or load the relevant `Pattern` object as-needed.

    The cache can be disabled by setting the `enable_cache` attribute to `False`.
    """
    dict: Dict[str, Callable[[], 'Pattern']]
    cache: Dict[str, 'Pattern']
    enable_cache: bool = True

    def __init__(self) -> None:
        self.dict = {}
        self.cache = {}

    def __setitem__(self, key: str, value: Callable[[], 'Pattern']) -> None:
        self.dict[key] = value
        if key in self.cache:
            del self.cache[key]

    def __delitem__(self, key: str) -> None:
        del self.dict[key]
        if key in self.cache:
            del self.cache[key]

    def __getitem__(self, key: str) -> 'Pattern':
        logger.debug(f'loading {key}')
        if self.enable_cache and key in self.cache:
            logger.debug(f'found {key} in cache')
            return self.cache[key]

        func = self.dict[key]
        pat = func()
        self.cache[key] = pat
        return pat

    def __iter__(self) -> Iterator[str]:
        return iter(self.dict)

    def __len__(self) -> int:
        return len(self.dict)

    def _set(self, key: str, value: 'Pattern') -> None:
        self[key] = lambda: value

    def _merge(self, other: Mapping[str, 'Pattern'], key: str) -> None:
        if isinstance(other, LazyLibrary):
            self.dict[key] = other.dict[key]
            if key in other.cache:
                self.cache[key] = other.cache[key]
        else:
            self._set(key, other[key])

    def __repr__(self) -> str:
        return '<LazyLibrary with keys ' + repr(list(self.keys())) + '>'

    def precache(self: LL) -> LL:
        """
        Force all patterns into the cache

        Returns:
            self
        """
        for key in self.dict:
            _ = self.dict.__getitem__(key)
        return self

    def clear_cache(self: LL) -> LL:
        """
        Clear the cache of this library.
        This is usually used before modifying or deleting patterns, e.g. when merging
          with another library.

        Returns:
            self
        """
        self.cache.clear()
        return self

    def __deepcopy__(self, memo: Optional[Dict] = None) -> 'LazyLibrary':
        raise LibraryError('LazyLibrary cannot be deepcopied (deepcopy doesn\'t descend into closures)')
