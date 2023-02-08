"""
Library classes for managing unique name->pattern mappings and
 deferred loading or creation.

# TODO documentn all library classes
# TODO toplevel documentation of library, classes, and abstracts
"""
from typing import List, Dict, Callable, TypeVar, Type, TYPE_CHECKING, cast
from typing import Tuple, Union, Iterator, Mapping, MutableMapping, Set, Optional, Sequence
import logging
import base64
import struct
import re
import copy
from pprint import pformat
from collections import defaultdict
from abc import ABCMeta, abstractmethod

import numpy
from numpy.typing import ArrayLike

from .error import LibraryError, PatternError
from .utils import rotation_matrix_2d, normalize_mirror
from .shapes import Shape, Polygon
from .label import Label
from .abstract import Abstract

if TYPE_CHECKING:
    from .pattern import Pattern, NamedPattern


logger = logging.getLogger(__name__)


visitor_function_t = Callable[..., 'Pattern']
L = TypeVar('L', bound='Library')
ML = TypeVar('ML', bound='MutableLibrary')
LL = TypeVar('LL', bound='LazyLibrary')


def _rename_patterns(lib: 'Library', name: str) -> str:
    # TODO document rename function
    if not name.startswith('_'):
        return name

    stem = name.split('$')[0]
    return lib.get_name(stem)


class Library(Mapping[str, 'Pattern'], metaclass=ABCMeta):
    # inherited abstract functions
    #def __getitem__(self, key: str) -> 'Pattern':
    #def __iter__(self) -> Iterator[str]:
    #def __len__(self) -> int:

    #__contains__, keys, items, values, get, __eq__, __ne__ supplied by Mapping

    def abstract_view(self) -> 'AbstractView':
        return AbstractView(self)

    def abstract(self, name: str) -> Abstract:
        """
        Return an `Abstract` (name & ports) for the pattern in question.

        Args:
            name: The pattern name

        Returns:
            An `Abstract` object for the pattern
        """
        return Abstract(name=name, ports=self[name].ports)

    def __repr__(self) -> str:
        return '<Library with keys\n' + pformat(list(self.keys())) + '>'

    def dangling_refs(
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
        if isinstance(tops, str):
            tops = (tops,)

        keep: Set[str] = self.referenced_patterns(tops) - set((None,))      # type: ignore
        keep |= set(tops)

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
            flatten_ports: bool = False,       # TODO document
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
                if not flatten_ports:
                    p.ports.clear()
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
            quiet: Optional[bool] = None,
            ) -> str:
        """
        Find a unique name for the pattern.

        This function may be overridden in a subclass or monkey-patched to fit the caller's requirements.

        Args:
            name: Preferred name for the pattern. Default '__'.
            sanitize: Allows only alphanumeric charaters and _?$. Replaces invalid characters with underscores.
            max_length: Names longer than this will be truncated.
            quiet: If `True`, suppress log messages. Default `None` suppresses messages only if
                the name starts with an underscore.

        Returns:
            Name, unique within this library.
        """
        if quiet is None:
            quiet = name.startswith('_')

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
            pattern: 'Pattern',
            visit_before: Optional[visitor_function_t] = None,
            visit_after: Optional[visitor_function_t] = None,
            *,
            hierarchy: Tuple[Optional[str], ...] = (None,),
            transform: Union[ArrayLike, bool, None] = False,
            memo: Optional[Dict] = None,
            ) -> L:
        """
        Convenience function.
        Performs a depth-first traversal of a pattern and its referenced patterns.
        At each pattern in the tree, the following sequence is called:
            ```
            current_pattern = visit_before(current_pattern, **vist_args)
            for sp in current_pattern.refs]
                self.dfs(sp.target, visit_before, visit_after,
                         hierarchy + (sp.target,), updated_transform, memo)
            current_pattern = visit_after(current_pattern, **visit_args)
            ```
          where `visit_args` are
            `hierarchy`:  (top_pattern_or_None, L1_pattern, L2_pattern, ..., parent_pattern)
                          tuple of all parent-and-higher pattern names. Top pattern name may be
                          `None` if not provided in first call to .dfs()
            `transform`:  numpy.ndarray containing cumulative
                          [x_offset, y_offset, rotation (rad), mirror_x (0 or 1)]
                          for the instance being visited
            `memo`:  Arbitrary dict (not altered except by `visit_before()` and `visit_after()`)

        Args:
            pattern: Pattern object to start at ("top"/root node of the tree).
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
                Default is (None,), which will be used as a placeholder for the top pattern's
                name if not overridden.

        Returns:
            self
        """
        if memo is None:
            memo = {}

        if transform is None or transform is True:
            transform = numpy.zeros(4)
        elif transform is not False:
            transform = numpy.array(transform)

        original_pattern = pattern

        if visit_before is not None:
            pattern = visit_before(pattern, hierarchy=hierarchy, memo=memo, transform=transform)

        for ref in pattern.refs:
            if transform is not False:
                sign = numpy.ones(2)
                if transform[3]:
                    sign[1] = -1
                xy = numpy.dot(rotation_matrix_2d(transform[2]), ref.offset * sign)
                mirror_x, angle = normalize_mirror(ref.mirrored)
                angle += ref.rotation
                ref_transform = transform + (xy[0], xy[1], angle, mirror_x)
                ref_transform[3] %= 2
            else:
                ref_transform = False

            if ref.target is None:
                continue
            if ref.target in hierarchy:
                raise LibraryError(f'.dfs() called on pattern with circular reference to "{ref.target}"')

            self.dfs(
                pattern=self[ref.target],
                visit_before=visit_before,
                visit_after=visit_after,
                hierarchy=hierarchy + (ref.target,),
                transform=ref_transform,
                memo=memo,
                )

        if visit_after is not None:
            pattern = visit_after(pattern, hierarchy=hierarchy, memo=memo, transform=transform)

        if pattern is not original_pattern:
            name = hierarchy[-1]
            if not isinstance(self, MutableLibrary):
                raise LibraryError('visit_* functions returned a new `Pattern` object'
                                   ' but the library is immutable')
            if name is None:
                raise LibraryError('visit_* functions returned a new `Pattern` object'
                                   ' but no top-level name was provided in `hierarchy`')

            cast(MutableLibrary, self)[name] = pattern

        return self


class MutableLibrary(Library, MutableMapping[str, 'Pattern'], metaclass=ABCMeta):
    # inherited abstract functions
    #def __getitem__(self, key: str) -> 'Pattern':
    #def __iter__(self) -> Iterator[str]:
    #def __len__(self) -> int:
    #def __setitem__(self, key: str, value: Union['Pattern', Callable[[], 'Pattern']]) -> None:
    #def __delitem__(self, key: str) -> None:

    @abstractmethod
    def __setitem__(
            self,
            key: str,
            value: Union['Pattern', Callable[[], 'Pattern']],
            ) -> None:
        pass

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        pass

    @abstractmethod
    def _merge(self, key_self: str, other: Mapping[str, 'Pattern'], key_other: str) -> None:
        pass

    def rename(
            self: ML,
            old_name: str,
            new_name: str,
            move_references: bool = False,
            ) -> ML:
        """
        Rename a pattern.

        Args:
            old_name: Current name for the pattern
            new_name: New name for the pattern
            #TODO move_Reference

        Returns:
            self
        """
        self[new_name] = self[old_name]
        del self[old_name]
        if move_references:
            self.move_references(old_name, new_name)
        return self

    def move_references(self: ML, old_target: str, new_target: str) -> ML:
        """
        Change all references pointing at `old_target` into references pointing at `new_target`.

        Args:
            old_target: Current reference target
            new_target: New target for the reference

        Returns:
            self
        """
        for pattern in self.values():
            for ref in pattern.refs:
                if ref.target == old_target:
                    ref.target = new_target
        return self

    def create(self, name: str) -> 'NamedPattern':
        """
        Convenience method to create an empty pattern, choose a name
        for it, add it with that name, and return both the pattern and name.

        Args:
            base_name: Prefix used when naming the pattern

        Returns:
            (name, pattern) tuple
        """
        from .pattern import NamedPattern
        #name = self.get_name(base_name)
        npat = NamedPattern(name)
        self[name] = npat
        return npat

    def set(
            self,
            name: str,
            value: Union['Pattern', Callable[[], 'Pattern']],
            ) -> str:
        """
        Convenience method which finds a suitable name for the provided
        pattern, adds it with that name, and returns the name.

        Args:
            base_name: Prefix used when naming the pattern
            value: The pattern (or callable used to generate it)

        Returns:
            The name of the pattern.
        """
        #name = self.get_name(base_name)
        self[name] = value
        return name

    def add(
            self,
            other: Mapping[str, 'Pattern'],
            rename_theirs: Callable[['Library', str], str] = _rename_patterns,
            ) -> Dict[str, str]:
        """
        Add keys from another library into this one.

        # TODO explain reference renaming and return

        Args:
            other: The library to insert keys from
            rename_theirs: Called as rename_theirs(self, name) for each duplicate name
                encountered in `other`. Should return the new name for the pattern in
                `other`.
                Default is effectively
                    `name.split('$')[0] if name.startswith('_') else name`
        Returns:
            self
        """
        duplicates = set(self.keys()) & set(other.keys())

        if not duplicates:
            for key in other.keys():
                self._merge(key, other, key)
            return {}

        temp = WrapLibrary(copy.deepcopy(dict(other)))      # TODO maybe add a `mutate` arg? Might want to keep the same patterns
        rename_map = {}
        for old_name in temp:
            if old_name in duplicates:
                new_name = rename_theirs(self, old_name)
                if new_name in self:
                    raise LibraryError(f'Unresolved duplicate key encountered in library merge: {old_name} -> {new_name}')
                rename_map[old_name] = new_name
            else:
                new_name = old_name

            self._merge(new_name, temp, old_name)

        # Update references in the newly-added cells
        for old_name in temp:
            new_name = rename_map.get(old_name, old_name)
            for ref in self[new_name].refs:
                ref.target = rename_map.get(ref.target, ref.target)

        return rename_map

    def add_tree(
            self,
            tree: 'Tree',
            name: Optional[str] = None,
            rename_theirs: Callable[['Library', str], str] = _rename_patterns,
            ) -> str:
        """
        Add a `Tree` object into the current library.

        Args:
            tree: The `Tree` object (a `Library` with a specified `top` Pattern)
                which will be added into the current library.
            name: New name for the top-level pattern. If not given, `tree.top` is used.
            rename_theirs: Called as rename_theirs(self, name) for each duplicate name
                encountered in `other`. Should return the new name for the pattern in
                `other`.
                Default is effectively
                    `name.split('$')[0] if name.startswith('_') else name`

        Returns:
            The new name for the top-level pattern (either `name` or `tree.top`).
        """
        if name is None:
            name = tree.top
        else:
            tree.library.rename(tree.top, name, move_references=True)
            tree.top = name

        rename_map = self.add(tree.library, rename_theirs=rename_theirs)
        return rename_map.get(name, name)

    def __lshift__(self, other: Mapping[str, 'Pattern']) -> str:
        if isinstance(other, Tree):
            return self.add_tree(other)     # Add library and return topcell name

        if len(other) != 1:
            raise LibraryError('Received a non-Tree library containing multiple cells')

        name = next(iter(other))
        self.add(other)
        return name

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
                return self.get_name('_shape')
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
            self[label2name(ll)] = pp

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
                return self.get_name('_rep')
            #name_func = lambda _pat, _shape: self.get_name('_rep')

        for pat in tuple(self.values()):
            new_shapes = []
            for shape in pat.shapes:
                if shape.repetition is None:
                    new_shapes.append(shape)
                    continue

                name = name_func(pat, shape)
                self[name] = Pattern(shapes=[shape])
                pat.ref(name, repetition=shape.repetition)
                shape.repetition = None
            pat.shapes = new_shapes

            new_labels = []
            for label in pat.labels:
                if label.repetition is None:
                    new_labels.append(label)
                    continue
                name = name_func(pat, label)
                self[name] = Pattern(labels=[label])
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
        if isinstance(tops, str):
            tops = (tops,)

        keep: Set[str] = self.referenced_patterns(tops) - set((None,))      # type: ignore
        keep |= set(tops)

        new = type(self)()
        for key in keep:
            new._merge(key, self, key)
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

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return f'<WrapROLibrary ({type(self.mapping)}) with keys\n' + pformat(list(self.keys())) + '>'


class WrapLibrary(MutableLibrary):
    mapping: MutableMapping[str, 'Pattern']

    def __init__(
            self,
            mapping: Optional[MutableMapping[str, 'Pattern']] = None,
            ) -> None:
        if mapping is None:
            self.mapping = {}
        else:
            self.mapping = mapping

    def __getitem__(self, key: str) -> 'Pattern':
        return self.mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def __setitem__(
            self,
            key: str,
            value: Union['Pattern', Callable[[], 'Pattern']],
            ) -> None:
        if key in self.mapping:
            raise LibraryError(f'"{key}" already exists in the library. Overwriting is not allowed!')

        if callable(value):
            value = value()
        elif hasattr(value, 'as_pattern'):
            value = cast('NamedPattern', value).as_pattern()      # don't want to carry along NamedPattern instances
        else:
            value = value
        self.mapping[key] = value

    def __delitem__(self, key: str) -> None:
        del self.mapping[key]

    def _merge(self, key_self: str, other: Mapping[str, 'Pattern'], key_other: str) -> None:
        self[key_self] = other[key_other]

    def __repr__(self) -> str:
        return f'<WrapLibrary ({type(self.mapping)}) with keys\n' + pformat(list(self.keys())) + '>'


class LazyLibrary(MutableLibrary):
    """
    This class is usually used to create a library of Patterns by mapping names to
     functions which generate or load the relevant `Pattern` object as-needed.
    """
    dict: Dict[str, Callable[[], 'Pattern']]
    cache: Dict[str, 'Pattern']
    _lookups_in_progress: Set[str]

    def __init__(self) -> None:
        self.dict = {}
        self.cache = {}
        self._lookups_in_progress = set()

    def __setitem__(
            self,
            key: str,
            value: Union['Pattern', Callable[[], 'Pattern']],
            ) -> None:
        if key in self.dict:
            raise LibraryError(f'"{key}" already exists in the library. Overwriting is not allowed!')

        if callable(value):
            value_func = value
        else:
            value_func = lambda: cast('Pattern', value)      # noqa: E731

        self.dict[key] = value_func
        if key in self.cache:
            del self.cache[key]

    def __delitem__(self, key: str) -> None:
        del self.dict[key]
        if key in self.cache:
            del self.cache[key]

    def __getitem__(self, key: str) -> 'Pattern':
        logger.debug(f'loading {key}')
        if key in self.cache:
            logger.debug(f'found {key} in cache')
            return self.cache[key]

        if key in self._lookups_in_progress:
            raise LibraryError(
                f'Detected multiple simultaneous lookups of "{key}".\n'
                'This may be caused by an invalid (cyclical) reference, or buggy code.\n'
                'If you are lazy-loading a file, try a non-lazy load and check for refernce cycles.'        # TODO give advice on finding cycles
                )

        self._lookups_in_progress.add(key)
        func = self.dict[key]
        pat = func()
        self._lookups_in_progress.remove(key)
        self.cache[key] = pat
        return pat

    def __iter__(self) -> Iterator[str]:
        return iter(self.dict)

    def __len__(self) -> int:
        return len(self.dict)

    def __contains__(self, key: object) -> bool:
        return key in self.dict

    def _merge(self, key_self: str, other: Mapping[str, 'Pattern'], key_other: str) -> None:
        if isinstance(other, LazyLibrary):
            self.dict[key_self] = other.dict[key_other]
            if key_other in other.cache:
                self.cache[key_self] = other.cache[key_other]
        else:
            self[key_self] = other[key_other]

    def __repr__(self) -> str:
        return '<LazyLibrary with keys\n' + pformat(list(self.keys())) + '>'

    def rename(
            self: LL,
            old_name: str,
            new_name: str,
            move_references: bool = False,
            ) -> LL:
        """
        Rename a pattern.

        Args:
            old_name: Current name for the pattern
            new_name: New name for the pattern
            move_references: Whether to scan all refs in the pattern and
                move them to point to `new_name` as necessary.
                Default `False`.

        Returns:
            self
        """
        self[new_name] = self.dict[old_name]        # copy over function
        if old_name in self.cache:
            self.cache[new_name] = self.cache[old_name]
        del self[old_name]

        if move_references:
            self.move_references(old_name, new_name)

        return self

    def move_references(self: LL, old_target: str, new_target: str) -> LL:
        """
        Change all references pointing at `old_target` into references pointing at `new_target`.

        Args:
            old_target: Current reference target
            new_target: New target for the reference

        Returns:
            self
        """
        self.precache()
        for pattern in self.cache.values():
            for ref in pattern.refs:
                if ref.target == old_target:
                    ref.target = new_target
        return self

    def precache(self: LL) -> LL:
        """
        Force all patterns into the cache

        Returns:
            self
        """
        for key in self.dict:
            _ = self[key]       # want to trigger our own __getitem__
        return self

    def __deepcopy__(self, memo: Optional[Dict] = None) -> 'LazyLibrary':
        raise LibraryError('LazyLibrary cannot be deepcopied (deepcopy doesn\'t descend into closures)')


class AbstractView(Mapping[str, Abstract]):
    library: Library

    def __init__(self, library: Library) -> None:
        self.library = library

    def __getitem__(self, key: str) -> Abstract:
        return self.library.abstract(key)

    def __iter__(self) -> Iterator[str]:
        return self.library.__iter__()

    def __len__(self) -> int:
        return self.library.__len__()


class Tree(MutableLibrary):
    top: str
    library: MutableLibrary

    @property
    def pattern(self) -> 'Pattern':
        return self.library[self.top]

    def __init__(
            self,
            top: Union[str, 'NamedPattern'],
            library: Optional[MutableLibrary] = None
            ) -> None:
        self.top = top if isinstance(top, str) else top.name
        self.library = library if library is not None else WrapLibrary()

    @classmethod
    def mk(cls, top: str) -> Tuple['Tree', 'Pattern']:
        from .pattern import Pattern
        tree = cls(top=top)
        pat = Pattern()
        tree[top] = pat
        return tree, pat

    def __getitem__(self, key: str) -> 'Pattern':
        return self.library[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.library)

    def __len__(self) -> int:
        return len(self.library)

    def __setitem__(self, key: str, value:  Union['Pattern', Callable[[], 'Pattern']]) -> None:
        self.library[key] = value

    def __delitem__(self, key: str) -> None:
        del self.library[key]

    def __repr__(self) -> str:
        return f'<Tree "{self.top}": {self.library} >'

    def _merge(self, key_self: str, other: Mapping[str, 'Pattern'], key_other: str) -> None:
        self.library._merge(key_self, other, key_other)
