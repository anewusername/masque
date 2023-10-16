"""
Library classes for managing unique name->pattern mappings and deferred loading or execution.

Classes include:
- `ILibraryView`: Defines a general interface for read-only name->pattern mappings.
- `LibraryView`: An implementation of `ILibraryView` backed by an arbitrary `Mapping`.
    Can be used to wrap any arbitrary `Mapping` to give it all the functionality in `ILibraryView`
- `ILibrary`: Defines a general interface for mutable name->pattern mappings.
- `Library`: An implementation of `ILibrary` backed by an arbitrary `MutableMapping`.
    Can be used to wrap any arbitrary `MutableMapping` to give it all the functionality in `ILibrary`.
    By default, uses a `dict` as the underylingmapping.
- `LazyLibrary`: An implementation of `ILibrary` which enables on-demand loading or generation
    of patterns.
- `AbstractView`: Provides a way to use []-indexing to generate abstracts for patterns in the linked
    library. Generated with `ILibraryView.abstract_view()`.
"""
from typing import Callable, Self, Type, TYPE_CHECKING, cast
from typing import Iterator, Mapping, MutableMapping, Sequence
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
from .utils import rotation_matrix_2d, layer_t
from .shapes import Shape, Polygon
from .label import Label
from .abstract import Abstract
from .pattern import map_layers

if TYPE_CHECKING:
    from .pattern import Pattern


logger = logging.getLogger(__name__)


visitor_function_t = Callable[..., 'Pattern']

SINGLE_USE_PREFIX = '_'
"""
Names starting with this prefix are assumed to refer to single-use patterns,
which may be renamed automatically by `ILibrary.add()` (via
`rename_theirs=_rename_patterns()` )
"""
# TODO what are the consequences of making '_' special?  maybe we can make this decision everywhere?


def _rename_patterns(lib: 'ILibraryView', name: str) -> str:
    """
    The default `rename_theirs` function for `ILibrary.add`.

      Treats names starting with `SINGLE_USE_PREFIX` (default: one underscore) as
    "one-offs" for which name conflicts should be automatically resolved.
    Conflicts are resolved by calling `lib.get_name(SINGLE_USE_PREFIX + stem)`
    where `stem = name.removeprefix(SINGLE_USE_PREFIX).split('$')[0]`.
    Names lacking the prefix are directly returned (not renamed).

    Args:
        lib: The library into which `name` is to be added (but is presumed to conflict)
        name: The original name, to be modified

    Returns:
        The new name, not guaranteed to be conflict-free!
    """
    if not name.startswith(SINGLE_USE_PREFIX):
        return name

    stem = name.removeprefix(SINGLE_USE_PREFIX).split('$')[0]
    return lib.get_name(SINGLE_USE_PREFIX + stem)


class ILibraryView(Mapping[str, 'Pattern'], metaclass=ABCMeta):
    """
    Interface for a read-only library.

    A library is a mapping from unique names (str) to collections of geometry (`Pattern`).
    """
    # inherited abstract functions
    #def __getitem__(self, key: str) -> 'Pattern':
    #def __iter__(self) -> Iterator[str]:
    #def __len__(self) -> int:

    #__contains__, keys, items, values, get, __eq__, __ne__ supplied by Mapping

    def __repr__(self) -> str:
        return '<ILibraryView with keys\n' + pformat(list(self.keys())) + '>'

    def abstract_view(self) -> 'AbstractView':
        """
        Returns:
            An AbstractView into this library
        """
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

    def dangling_refs(
            self,
            tops: str | Sequence[str] | None = None,
            ) -> set[str | None]:
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
            tops: str | Sequence[str] | None = None,
            skip: set[str | None] | None = None,
            ) -> set[str | None]:
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
            tops: str | Sequence[str],
            ) -> 'ILibraryView':
        """
         Return a new `ILibraryView`, containing only the specified patterns and the patterns they
        reference (recursively).
        Dangling references do not cause an error.

        Args:
            tops: Name(s) of patterns to keep

        Returns:
            A `LibraryView` containing only `tops` and the patterns they reference.
        """
        if isinstance(tops, str):
            tops = (tops,)

        keep = cast(set[str], self.referenced_patterns(tops) - set((None,)))
        keep |= set(tops)

        filtered = {kk: vv for kk, vv in self.items() if kk in keep}
        new = LibraryView(filtered)
        return new

    def polygonize(
            self,
            num_vertices: int | None = None,
            max_arclen: float | None = None,
            ) -> Self:
        """
        Calls `.polygonize(...)` on each pattern in this library.
        Arguments are passed on to `shape.to_polygons(...)`.

        Args:
            num_vertices: Number of points to use for each polygon. Can be overridden by
                `max_arclen` if that results in more points. Optional, defaults to shapes'
                internal defaults.
            max_arclen: Maximum arclength which can be approximated by a single line
             segment. Optional, defaults to shapes' internal defaults.

        Returns:
            self
        """
        for pat in self.values():
            pat.polygonize(num_vertices, max_arclen)
        return self

    def manhattanize(
            self,
            grid_x: ArrayLike,
            grid_y: ArrayLike,
            ) -> Self:
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
            tops: str | Sequence[str],
            flatten_ports: bool = False,
            ) -> dict[str, 'Pattern']:
        """
        Returns copies of all `tops` patterns with all refs
         removed and replaced with equivalent shapes.
        Also returns flattened copies of all referenced patterns.
        The originals in the calling `Library` are not modified.
        For an in-place variant, see `Pattern.flatten`.

        Args:
            tops: The pattern(s) to flattern.
            flatten_ports: If `True`, keep ports from any referenced
                patterns; otherwise discard them.

        Returns:
            {name: flat_pattern} mapping for all flattened patterns.
        """
        if isinstance(tops, str):
            tops = (tops,)

        flattened: dict[str, 'Pattern | None'] = {}

        def flatten_single(name: str) -> None:
            flattened[name] = None
            pat = self[name].deepcopy()

            for target in pat.refs:
                if target is None:
                    continue
                if target not in flattened:
                    flatten_single(target)

                target_pat = flattened[target]
                if target_pat is None:
                    raise PatternError(f'Circular reference in {name} to {target}')
                if target_pat.is_empty():        # avoid some extra allocations
                    continue

                for ref in pat.refs[target]:
                    p = ref.as_pattern(pattern=target_pat)
                    if not flatten_ports:
                        p.ports.clear()
                    pat.append(p)

            pat.refs.clear()
            flattened[name] = pat

        for top in tops:
            flatten_single(top)

        assert None not in flattened.values()
        return cast(dict[str, 'Pattern'], flattened)

    def get_name(
            self,
            name: str = SINGLE_USE_PREFIX * 2,
            sanitize: bool = True,
            max_length: int = 32,
            quiet: bool | None = None,
            ) -> str:
        """
        Find a unique name for the pattern.

        This function may be overridden in a subclass or monkey-patched to fit the caller's requirements.

        Args:
            name: Preferred name for the pattern. Default is `SINGLE_USE_PREFIX * 2`.
            sanitize: Allows only alphanumeric charaters and _?$. Replaces invalid characters with underscores.
            max_length: Names longer than this will be truncated.
            quiet: If `True`, suppress log messages. Default `None` suppresses messages only if
                the name starts with `SINGLE_USE_PREFIX`.

        Returns:
            Name, unique within this library.
        """
        if quiet is None:
            quiet = name.startswith(SINGLE_USE_PREFIX)

        if sanitize:
            # Remove invalid characters
            sanitized_name = re.compile(r'[^A-Za-z0-9_\?\$]').sub('_', name)
        else:
            sanitized_name = name

        ii = 0
        suffixed_name = sanitized_name
        while suffixed_name in self or suffixed_name == '':
            suffix = base64.b64encode(struct.pack('>Q', ii), altchars=b'$?').decode('ASCII')

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

    def tops(self) -> list[str]:
        """
        Return the list of all patterns that are not referenced by any other pattern in the library.

        Returns:
            A list of pattern names in which no pattern is referenced by any other pattern.
        """
        names = set(self.keys())
        not_toplevel: set[str | None] = set()
        for name in names:
            not_toplevel |= set(self[name].refs.keys())

        toplevel = list(names - not_toplevel)
        return toplevel

    def top(self) -> str:
        """
        Return the name of the topcell, or raise an exception if there isn't a single topcell
        """
        tops = self.tops()
        if len(tops) != 1:
            raise LibraryError(f'Asked for the single topcell, but found the following: {pformat(tops)}')
        return tops[0]

    def top_pattern(self) -> 'Pattern':
        """
        Shorthand for self[self.top()]
        """
        return self[self.top()]

    def dfs(
            self,
            pattern: 'Pattern',
            visit_before: visitor_function_t | None = None,
            visit_after: visitor_function_t | None = None,
            *,
            hierarchy: tuple[str | None, ...] = (None,),
            transform: ArrayLike | bool | None = False,
            memo: dict | None = None,
            ) -> Self:
        """
        Convenience function.
        Performs a depth-first traversal of a pattern and its referenced patterns.
        At each pattern in the tree, the following sequence is called:
            ```
            current_pattern = visit_before(current_pattern, **vist_args)
            for target in current_pattern.refs:
                for ref in pattern.refs[target]:
                    self.dfs(target, visit_before, visit_after,
                             hierarchy + (sp.target,), updated_transform, memo)
            current_pattern = visit_after(current_pattern, **visit_args)
            ```
          where `visit_args` are
            `hierarchy`:  (top_pattern_or_None, L1_pattern, L2_pattern, ..., parent_pattern, target_pattern)
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

        for target in pattern.refs:
            if target is None:
                continue
            if target in hierarchy:
                raise LibraryError(f'.dfs() called on pattern with circular reference to "{target}"')

            for ref in pattern.refs[target]:
                if transform is not False:
                    sign = numpy.ones(2)
                    if transform[3]:
                        sign[1] = -1
                    xy = numpy.dot(rotation_matrix_2d(transform[2]), ref.offset * sign)
                    ref_transform = transform + (xy[0], xy[1], ref.rotation, ref.mirrored)
                    ref_transform[3] %= 2
                else:
                    ref_transform = False

                self.dfs(
                    pattern=self[target],
                    visit_before=visit_before,
                    visit_after=visit_after,
                    hierarchy=hierarchy + (target,),
                    transform=ref_transform,
                    memo=memo,
                    )

        if visit_after is not None:
            pattern = visit_after(pattern, hierarchy=hierarchy, memo=memo, transform=transform)

        if pattern is not original_pattern:
            name = hierarchy[-1]
            if not isinstance(self, ILibrary):
                raise LibraryError('visit_* functions returned a new `Pattern` object'
                                   ' but the library is immutable')
            if name is None:
                # The top pattern is not the original pattern, but we don't know what to call it!
                raise LibraryError('visit_* functions returned a new `Pattern` object'
                                   ' but no top-level name was provided in `hierarchy`')

            cast(ILibrary, self)[name] = pattern

        return self


class ILibrary(ILibraryView, MutableMapping[str, 'Pattern'], metaclass=ABCMeta):
    """
    Interface for a writeable library.

    A library is a mapping from unique names (str) to collections of geometry (`Pattern`).
    """
    # inherited abstract functions
    #def __getitem__(self, key: str) -> 'Pattern':
    #def __iter__(self) -> Iterator[str]:
    #def __len__(self) -> int:
    #def __setitem__(self, key: str, value: 'Pattern | Callable[[], Pattern]') -> None:
    #def __delitem__(self, key: str) -> None:

    @abstractmethod
    def __setitem__(
            self,
            key: str,
            value: 'Pattern | Callable[[], Pattern]',
            ) -> None:
        pass

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        pass

    @abstractmethod
    def _merge(self, key_self: str, other: Mapping[str, 'Pattern'], key_other: str) -> None:
        pass

    def rename(
            self,
            old_name: str,
            new_name: str,
            move_references: bool = False,
            ) -> Self:
        """
        Rename a pattern.

        Args:
            old_name: Current name for the pattern
            new_name: New name for the pattern
            move_references: If `True`, any refs in this library pointing to `old_name`
                will be updated to point to `new_name`.

        Returns:
            self
        """
        self[new_name] = self[old_name]
        del self[old_name]
        if move_references:
            self.move_references(old_name, new_name)
        return self

    def rename_top(self, name: str) -> Self:
        """
        Rename the (single) top pattern
        """
        self.rename(self.top(), name, move_references=True)
        return self

    def move_references(self, old_target: str, new_target: str) -> Self:
        """
        Change all references pointing at `old_target` into references pointing at `new_target`.

        Args:
            old_target: Current reference target
            new_target: New target for the reference

        Returns:
            self
        """
        for pattern in self.values():
            if old_target in pattern.refs:
                pattern.refs[new_target].extend(pattern.refs[old_target])
                del pattern.refs[old_target]
        return self

    def map_layers(
            self,
            map_layer: Callable[[layer_t], layer_t],
            ) -> Self:
        """
        Move all the elements in all patterns from one layer onto a different layer.
        Can also handle multiple such mappings simultaneously.

        Args:
            map_layer: Callable which may be called with each layer present in `elements`,
                and should return the new layer to which it will be mapped.
                A simple example which maps `old_layer` to `new_layer` and leaves all others
                as-is would look like `lambda layer: {old_layer: new_layer}.get(layer, layer)`

        Returns:
            self
        """
        for pattern in self.values():
            pattern.shapes = map_layers(pattern.shapes, map_layer)
            pattern.labels = map_layers(pattern.labels, map_layer)
        return self

    def mkpat(self, name: str) -> tuple[str, 'Pattern']:
        """
        Convenience method to create an empty pattern, add it to the library,
        and return both the pattern and name.

        Args:
            name: Name for the pattern

        Returns:
            (name, pattern) tuple
        """
        from .pattern import Pattern
        pat = Pattern()
        self[name] = pat
        return name, pat

    def add(
            self,
            other: Mapping[str, 'Pattern'],
            rename_theirs: Callable[['ILibraryView', str], str] = _rename_patterns,
            mutate_other: bool = False,
            ) -> dict[str, str]:
        """
        Add items from another library into this one.

        If any name in `other` is already present in `self`, `rename_theirs(self, name)` is called
          to pick a new name for the newly-added pattern. If the new name still conflicts with a name
          in `self` a `LibraryError` is raised. All references to the original name (within `other)`
          are updated to the new name.
        If `mutate_other=False` (default), all changes are made to a deepcopy of `other`.

        By default, `rename_theirs` makes no changes to the name (causing a `LibraryError`) unless the
          name starts with `SINGLE_USE_PREFIX`. Prefixed names are truncated to before their first
          non-prefix '$' and then passed to `self.get_name()` to create a new unique name.

        Args:
            other: The library to insert keys from.
            rename_theirs: Called as rename_theirs(self, name) for each duplicate name
                encountered in `other`. Should return the new name for the pattern in
                `other`. See above for default behavior.
            mutate_other: If `True`, modify the original library and its contained patterns
                (e.g. when renaming patterns and updating refs). Otherwise, operate on a deepcopy
                (default).

        Returns:
            A mapping of `{old_name: new_name}` for all `old_name`s in `other`. Unchanged
            names map to themselves.

        Raises:
            `LibraryError` if a duplicate name is encountered even after applying `rename_theirs()`.
        """
        from .pattern import map_targets
        duplicates = set(self.keys()) & set(other.keys())

        if not duplicates:
            for key in other.keys():
                self._merge(key, other, key)
            return {}

        if mutate_other:
            if isinstance(other, Library):
                temp = other
            else:
                temp = Library(dict(other))
        else:
            temp = Library(copy.deepcopy(dict(other)))
        rename_map = {}
        for old_name in temp:
            if old_name in self:
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
            pat = self[new_name]
            pat.refs = map_targets(pat.refs, lambda tt: cast(dict[str | None, str | None], rename_map).get(tt, tt))

        return rename_map

    def __lshift__(self, other: Mapping[str, 'Pattern']) -> str:
        if len(other) == 1:
            name = next(iter(other))
        else:
            if not isinstance(other, ILibraryView):
                other = LibraryView(other)

            tops = other.tops()
            if len(tops) > 1:
                raise LibraryError('Received a library containing multiple topcells!')

            name = tops[0]

        rename_map = self.add(other)
        new_name = rename_map.get(name, name)
        return new_name

    def __le__(self, other: Mapping[str, 'Pattern']) -> Abstract:
        new_name = self << other
        return self.abstract(new_name)

    def dedup(
            self,
            norm_value: int = int(1e6),
            exclude_types: tuple[Type] = (Polygon,),
            label2name: Callable[[tuple], str] | None = None,
            threshold: int = 2,
            ) -> Self:
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
                a name for the generated pattern.
                Default `self.get_name(SINGLE_USE_PREIX + 'shape')`.
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
                return self.get_name(SINGLE_USE_PREFIX + 'shape')

        shape_counts: MutableMapping[tuple, int] = defaultdict(int)
        shape_funcs = {}

        # ## First pass ##
        #  Using the label tuple from `.normalized_form()` as a key, check how many of each shape
        # are present and store the shape function for each one
        for pat in tuple(self.values()):
            for layer, sseq in pat.shapes.items():
                for shape in sseq:
                    if not any(isinstance(shape, t) for t in exclude_types):
                        base_label, _values, func = shape.normalized_form(norm_value)
                        label = (*base_label, layer)
                        shape_funcs[label] = func
                        shape_counts[label] += 1

        shape_pats = {}
        for label, count in shape_counts.items():
            if count < threshold:
                continue

            shape_func = shape_funcs[label]
            shape_pat = Pattern()
            shape_pat.shapes[label[-1]] += [shape_func()]
            shape_pats[label] = shape_pat

        # ## Second pass ##
        for pat in tuple(self.values()):
            #  Store `[(index_in_shapes, values_from_normalized_form), ...]` for all shapes which
            # are to be replaced.
            # The `values` are `(offset, scale, rotation)`.

            shape_table: dict[tuple, list] = defaultdict(list)
            for layer, sseq in pat.shapes.items():
                for i, shape in enumerate(sseq):
                    if any(isinstance(shape, t) for t in exclude_types):
                        continue

                    base_label, values, _func = shape.normalized_form(norm_value)
                    label = (*base_label, layer)

                    if label not in shape_pats:
                        continue

                    shape_table[label].append((i, values))

            #  For repeated shapes, create a `Pattern` holding a normalized shape object,
            # and add `pat.refs` entries for each occurrence in pat. Also, note down that
            # we should delete the `pat.shapes` entries for which we made `Ref`s.
            shapes_to_remove = []
            for label in shape_table:
                layer = label[-1]
                target = label2name(label)
                for ii, values in shape_table[label]:
                    offset, scale, rotation, mirror_x = values
                    pat.ref(target=target, offset=offset, scale=scale,
                            rotation=rotation, mirrored=(mirror_x, False))
                    shapes_to_remove.append(ii)

                # Remove any shapes for which we have created refs.
                for ii in sorted(shapes_to_remove, reverse=True):
                    del pat.shapes[layer][ii]

        for ll, pp in shape_pats.items():
            self[label2name(ll)] = pp

        return self

    def wrap_repeated_shapes(
            self,
            name_func: Callable[['Pattern', Shape | Label], str] | None = None,
            ) -> Self:
        """
        Wraps all shapes and labels with a non-`None` `repetition` attribute
          into a `Ref`/`Pattern` combination, and applies the `repetition`
          to each `Ref` instead of its contained shape.

        Args:
            name_func: Function f(this_pattern, shape) which generates a name for the
                        wrapping pattern.
                        Default is `self.get_name(SINGLE_USE_PREFIX + 'rep')`.

        Returns:
            self
        """
        from .pattern import Pattern

        if name_func is None:
            def name_func(_pat, _shape):
                return self.get_name(SINGLE_USE_PREFIX = 'rep')

        for pat in tuple(self.values()):
            for layer in pat.shapes:
                new_shapes = []
                for shape in pat.shapes[layer]:
                    if shape.repetition is None:
                        new_shapes.append(shape)
                        continue

                    name = name_func(pat, shape)
                    self[name] = Pattern(shapes={layer: [shape]})
                    pat.ref(name, repetition=shape.repetition)
                    shape.repetition = None
                pat.shapes[layer] = new_shapes

            for layer in pat.labels:
                new_labels = []
                for label in pat.labels[layer]:
                    if label.repetition is None:
                        new_labels.append(label)
                        continue
                    name = name_func(pat, label)
                    self[name] = Pattern(labels={layer: [label]})
                    pat.ref(name, repetition=label.repetition)
                    label.repetition = None
                pat.labels[layer] = new_labels

        return self

    def subtree(
            self,
            tops: str | Sequence[str],
            ) -> Self:
        """
         Return a new `ILibraryView`, containing only the specified patterns and the patterns they
        reference (recursively).
        Dangling references do not cause an error.

        Args:
            tops: Name(s) of patterns to keep

        Returns:
            An object of the same type as `self` containing only `tops` and the patterns they reference.
        """
        if isinstance(tops, str):
            tops = (tops,)

        keep = cast(set[str], self.referenced_patterns(tops) - set((None,)))
        keep |= set(tops)

        new = type(self)()
        for key in keep & set(self.keys()):
            new._merge(key, self, key)
        return new

    def prune_empty(
            self,
            repeat: bool = True,
            ) -> set[str]:
        """
        Delete any empty patterns (i.e. where `Pattern.is_empty` returns `True`).

        Args:
            repeat: Also recursively delete any patterns which only contain(ed) empty patterns.

        Returns:
            A set containing the names of all deleted patterns
        """
        trimmed = set()
        while empty := set(name for name, pat in self.items() if pat.is_empty()):
            for name in empty:
                del self[name]

            for pat in self.values():
                for name in empty:
                    # Second pass to skip looking at refs in empty patterns
                    if name in pat.refs:
                        del pat.refs[name]

            trimmed |= empty
            if not repeat:
                break
        return trimmed

    def delete(
            self,
            key: str,
            delete_refs: bool = True,
            ) -> Self:
        """
        Delete a pattern and (optionally) all refs pointing to that pattern.

        Args:
            key: Name of the pattern to be deleted.
            delete_refs: If `True` (default), also delete all refs pointing to the pattern.
        """
        del self[key]
        if delete_refs:
            for pat in self.values():
                if key in pat.refs:
                    del pat.refs[key]
        return self


class LibraryView(ILibraryView):
    """
    Default implementation for a read-only library.

    A library is a mapping from unique names (str) to collections of geometry (`Pattern`).
    This library is backed by an arbitrary python object which implements the `Mapping` interface.
    """
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
        return f'<LibraryView ({type(self.mapping)}) with keys\n' + pformat(list(self.keys())) + '>'


class Library(ILibrary):
    """
    Default implementation for a writeable library.

    A library is a mapping from unique names (str) to collections of geometry (`Pattern`).
    This library is backed by an arbitrary python object which implements the `MutableMapping` interface.
    """
    mapping: MutableMapping[str, 'Pattern']

    def __init__(
            self,
            mapping: MutableMapping[str, 'Pattern'] | None = None,
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
            value: 'Pattern | Callable[[], Pattern]',
            ) -> None:
        if key in self.mapping:
            raise LibraryError(f'"{key}" already exists in the library. Overwriting is not allowed!')

        if callable(value):
            value = value()
        else:
            value = value
        self.mapping[key] = value

    def __delitem__(self, key: str) -> None:
        del self.mapping[key]

    def _merge(self, key_self: str, other: Mapping[str, 'Pattern'], key_other: str) -> None:
        self[key_self] = other[key_other]

    def __repr__(self) -> str:
        return f'<Library ({type(self.mapping)}) with keys\n' + pformat(list(self.keys())) + '>'

    @classmethod
    def mktree(cls, name: str) -> tuple[Self, 'Pattern']:
        """
        Create a new Library and immediately add a pattern

        Args:
            name: The name for the new pattern (usually the name of the topcell).

        Returns:
            The newly created `Library` and the newly created `Pattern`
        """
        from .pattern import Pattern
        tree = cls()
        pat = Pattern()
        tree[name] = pat
        return tree, pat


class LazyLibrary(ILibrary):
    """
    This class is usually used to create a library of Patterns by mapping names to
     functions which generate or load the relevant `Pattern` object as-needed.

    TODO: lots of stuff causes recursive loads (e.g. data_to_ports?). What should you avoid?
    """
    mapping: dict[str, Callable[[], 'Pattern']]
    cache: dict[str, 'Pattern']
    _lookups_in_progress: set[str]

    def __init__(self) -> None:
        self.mapping = {}
        self.cache = {}
        self._lookups_in_progress = set()

    def __setitem__(
            self,
            key: str,
            value: 'Pattern | Callable[[], Pattern]',
            ) -> None:
        if key in self.mapping:
            raise LibraryError(f'"{key}" already exists in the library. Overwriting is not allowed!')

        if callable(value):
            value_func = value
        else:
            value_func = lambda: cast('Pattern', value)      # noqa: E731

        self.mapping[key] = value_func
        if key in self.cache:
            del self.cache[key]

    def __delitem__(self, key: str) -> None:
        del self.mapping[key]
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
                'If you are lazy-loading a file, try a non-lazy load and check for reference cycles.'        # TODO give advice on finding cycles
                )

        self._lookups_in_progress.add(key)
        func = self.mapping[key]
        pat = func()
        self._lookups_in_progress.remove(key)
        self.cache[key] = pat
        return pat

    def __iter__(self) -> Iterator[str]:
        return iter(self.mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __contains__(self, key: object) -> bool:
        return key in self.mapping

    def _merge(self, key_self: str, other: Mapping[str, 'Pattern'], key_other: str) -> None:
        if isinstance(other, LazyLibrary):
            self.mapping[key_self] = other.mapping[key_other]
            if key_other in other.cache:
                self.cache[key_self] = other.cache[key_other]
        else:
            self[key_self] = other[key_other]

    def __repr__(self) -> str:
        return '<LazyLibrary with keys\n' + pformat(list(self.keys())) + '>'

    def rename(
            self,
            old_name: str,
            new_name: str,
            move_references: bool = False,
            ) -> Self:
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
        self[new_name] = self.mapping[old_name]        # copy over function
        if old_name in self.cache:
            self.cache[new_name] = self.cache[old_name]
        del self[old_name]

        if move_references:
            self.move_references(old_name, new_name)

        return self

    def move_references(self, old_target: str, new_target: str) -> Self:
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
            if old_target in pattern.refs:
                pattern.refs[new_target].extend(pattern.refs[old_target])
                del pattern.refs[old_target]
        return self

    def precache(self) -> Self:
        """
        Force all patterns into the cache

        Returns:
            self
        """
        for key in self.mapping:
            _ = self[key]       # want to trigger our own __getitem__
        return self

    def __deepcopy__(self, memo: dict | None = None) -> 'LazyLibrary':
        raise LibraryError('LazyLibrary cannot be deepcopied (deepcopy doesn\'t descend into closures)')


class AbstractView(Mapping[str, Abstract]):
    """
    A read-only mapping from names to `Abstract` objects.

    This is usually just used as a shorthand for repeated calls to `library.abstract()`.
    """
    library: ILibraryView

    def __init__(self, library: ILibraryView) -> None:
        self.library = library

    def __getitem__(self, key: str) -> Abstract:
        return self.library.abstract(key)

    def __iter__(self) -> Iterator[str]:
        return self.library.__iter__()

    def __len__(self) -> int:
        return self.library.__len__()
