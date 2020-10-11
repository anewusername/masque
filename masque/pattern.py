"""
 Base object representing a lithography mask.
"""

from typing import List, Callable, Tuple, Dict, Union, Set, Sequence, Optional, Type, overload
from typing import MutableMapping, Iterable
import copy
import pickle
from itertools import chain
from collections import defaultdict

import numpy        # type: ignore
from numpy import inf
# .visualize imports matplotlib and matplotlib.collections

from .subpattern import SubPattern
from .shapes import Shape, Polygon
from .label import Label
from .utils import rotation_matrix_2d, vector2, normalize_mirror, AutoSlots, annotations_t
from .error import PatternError, PatternLockedError
from .traits import LockableImpl, AnnotatableImpl, Scalable


visitor_function_t = Callable[['Pattern', Tuple['Pattern'], Dict, numpy.ndarray], 'Pattern']


class Pattern(LockableImpl, AnnotatableImpl, metaclass=AutoSlots):
    """
    2D layout consisting of some set of shapes, labels, and references to other Pattern objects
     (via SubPattern). Shapes are assumed to inherit from masque.shapes.Shape or provide equivalent functions.
    """
    __slots__ = ('shapes', 'labels', 'subpatterns', 'name')

    shapes: List[Shape]
    """ List of all shapes in this Pattern.
    Elements in this list are assumed to inherit from Shape or provide equivalent functions.
    """

    labels: List[Label]
    """ List of all labels in this Pattern. """

    subpatterns: List[SubPattern]
    """ List of all references to other patterns (`SubPattern`s) in this `Pattern`.
    Multiple objects in this list may reference the same Pattern object
      (i.e. multiple instances of the same object).
    """

    name: str
    """ A name for this pattern """

    def __init__(self,
                 name: str = '',
                 *,
                 shapes: Sequence[Shape] = (),
                 labels: Sequence[Label] = (),
                 subpatterns: Sequence[SubPattern] = (),
                 annotations: Optional[annotations_t] = None,
                 locked: bool = False,
                 ):
        """
        Basic init; arguments get assigned to member variables.
         Non-list inputs for shapes and subpatterns get converted to lists.

        Args:
            shapes: Initial shapes in the Pattern
            labels: Initial labels in the Pattern
            subpatterns: Initial subpatterns in the Pattern
            name: An identifier for the Pattern
            locked: Whether to lock the pattern after construction
        """
        LockableImpl.unlock(self)
        if isinstance(shapes, list):
            self.shapes = shapes
        else:
            self.shapes = list(shapes)

        if isinstance(labels, list):
            self.labels = labels
        else:
            self.labels = list(labels)

        if isinstance(subpatterns, list):
            self.subpatterns = subpatterns
        else:
            self.subpatterns = list(subpatterns)

        self.annotations = annotations if annotations is not None else {}
        self.name = name
        self.set_locked(locked)

    def __setattr__(self, name, value):
        if self.locked and name != 'locked':
            raise PatternLockedError()
        object.__setattr__(self, name, value)

    def  __copy__(self, memo: Dict = None) -> 'Pattern':
        return Pattern(name=self.name,
                       shapes=copy.deepcopy(self.shapes),
                       labels=copy.deepcopy(self.labels),
                       subpatterns=[copy.copy(sp) for sp in self.subpatterns],
                       annotations=copy.deepcopy(self.annotations),
                       locked=self.locked)

    def  __deepcopy__(self, memo: Dict = None) -> 'Pattern':
        memo = {} if memo is None else memo
        new = Pattern(name=self.name,
                shapes=copy.deepcopy(self.shapes, memo),
                labels=copy.deepcopy(self.labels, memo),
                subpatterns=copy.deepcopy(self.subpatterns, memo),
                annotations=copy.deepcopy(self.annotations, memo),
                locked=self.locked)
        return new

    def rename(self, name: str) -> 'Pattern':
        self.name = name
        return self

    def append(self, other_pattern: 'Pattern') -> 'Pattern':
        """
        Appends all shapes, labels and subpatterns from other_pattern to self's shapes,
          labels, and supbatterns.

        Args:
           other_pattern: The Pattern to append

        Returns:
            self
        """
        self.subpatterns += other_pattern.subpatterns
        self.shapes += other_pattern.shapes
        self.labels += other_pattern.labels
        return self

    def subset(self,
               shapes_func: Callable[[Shape], bool] = None,
               labels_func: Callable[[Label], bool] = None,
               subpatterns_func: Callable[[SubPattern], bool] = None,
               recursive: bool = False,
               ) -> 'Pattern':
        """
        Returns a Pattern containing only the entities (e.g. shapes) for which the
          given entity_func returns True.
        Self is _not_ altered, but shapes, labels, and subpatterns are _not_ copied.

        Args:
            shapes_func: Given a shape, returns a boolean denoting whether the shape is a member
                of the subset. Default always returns False.
            labels_func: Given a label, returns a boolean denoting whether the label is a member
                of the subset. Default always returns False.
            subpatterns_func: Given a subpattern, returns a boolean denoting if it is a member
                of the subset. Default always returns False.
            recursive: If True, also calls .subset() recursively on patterns referenced by this
                pattern.

        Returns:
            A Pattern containing all the shapes and subpatterns for which the parameter
                functions return True
        """
        def do_subset(src: Optional['Pattern']) -> Optional['Pattern']:
            if src is None:
                return None
            pat = Pattern(name=src.name)
            if shapes_func is not None:
                pat.shapes = [s for s in src.shapes if shapes_func(s)]
            if labels_func is not None:
                pat.labels = [s for s in src.labels if labels_func(s)]
            if subpatterns_func is not None:
                pat.subpatterns = [s for s in src.subpatterns if subpatterns_func(s)]
            return pat

        if recursive:
            pat = self.apply(do_subset)
        else:
            pat = do_subset(self)

        assert(pat is not None)
        return pat

    def apply(self,
              func: Callable[[Optional['Pattern']], Optional['Pattern']],
              memo: Optional[Dict[int, Optional['Pattern']]] = None,
              ) -> Optional['Pattern']:
        """
        Recursively apply func() to this pattern and any pattern it references.
        func() is expected to take and return a Pattern.
        func() is first applied to the pattern as a whole, then any referenced patterns.
        It is only applied to any given pattern once, regardless of how many times it is
            referenced.

        Args:
            func: Function which accepts a Pattern, and returns a pattern.
            memo: Dictionary used to avoid re-running on multiply-referenced patterns.
                Stores `{id(pattern): func(pattern)}` for patterns which have already been processed.
                Default `None` (no already-processed patterns).

        Returns:
            The result of applying func() to this pattern and all subpatterns.

        Raises:
            PatternError if called on a pattern containing a circular reference.
        """
        if memo is None:
            memo = {}

        pat_id = id(self)
        if pat_id not in memo:
            memo[pat_id] = None
            pat = func(self)
            if pat is not None:
                for subpat in pat.subpatterns:
                    if subpat.pattern is None:
                        subpat.pattern = func(None)
                    else:
                        subpat.pattern = subpat.pattern.apply(func, memo)
            memo[pat_id] = pat
        elif memo[pat_id] is None:
            raise PatternError('.apply() called on pattern with circular reference')
        else:
            pat = memo[pat_id]
        return pat

    def dfs(self,
            visit_before: visitor_function_t = None,
            visit_after: visitor_function_t = None,
            transform: Union[numpy.ndarray, bool, None] = False,
            memo: Optional[Dict] = None,
            hierarchy: Tuple['Pattern', ...] = (),
            ) -> 'Pattern':
        """
        Experimental convenience function.
        Performs a depth-first traversal of this pattern and its subpatterns.
        At each pattern in the tree, the following sequence is called:
            ```
            current_pattern = visit_before(current_pattern, **vist_args)
            for sp in current_pattern.subpatterns]
                sp.pattern = sp.pattern.df(visit_before, visit_after, updated_transform,
                                           memo, (current_pattern,) + hierarchy)
            current_pattern = visit_after(current_pattern, **visit_args)
            ```
          where `visit_args` are
            `hierarchy`:  (top_pattern, L1_pattern, L2_pattern, ..., parent_pattern)
                          tuple of all parent-and-higher patterns
            `transform`:  numpy.ndarray containing cumulative
                          [x_offset, y_offset, rotation (rad), mirror_x (0 or 1)]
                          for the instance being visited
            `memo`:  Arbitrary dict (not altered except by visit_*())

        Args:
            visit_before: Function to call before traversing subpatterns.
                Should accept a `Pattern` and `**visit_args`, and return the (possibly modified)
                pattern. Default `None` (not called).
            visit_after: Function to call after traversing subpatterns.
                Should accept a Pattern and **visit_args, and return the (possibly modified)
                pattern. Default `None` (not called).
            transform: Initial value for `visit_args['transform']`.
                Can be `False`, in which case the transform is not calculated.
                `True` or `None` is interpreted as `[0, 0, 0, 0]`.
            memo: Arbitrary dict for use by `visit_*()` functions. Default `None` (empty dict).
            hierarchy: Tuple of patterns specifying the hierarchy above the current pattern.
                Appended to the start of the generated `visit_args['hierarchy']`.
                Default is an empty tuple.

        Returns:
            The result, including `visit_before(self, ...)` and `visit_after(self, ...)`.
            Note that `self` may also be altered!
        """
        if memo is None:
            memo = {}

        if transform is None or transform is True:
            transform = numpy.zeros(4)

        if self in hierarchy:
            raise PatternError('.dfs() called on pattern with circular reference')

        pat = self
        if visit_before is not None:
            pat = visit_before(pat, hierarchy=hierarchy, memo=memo, transform=transform)        # type: ignore

        for subpattern in self.subpatterns:
            if transform is not False:
                sign = numpy.ones(2)
                if transform[3]:
                   sign[1] = -1
                xy = numpy.dot(rotation_matrix_2d(transform[2]), subpattern.offset * sign)
                mirror_x, angle = normalize_mirror(subpattern.mirrored)
                angle += subpattern.rotation
                sp_transform = transform + (xy[0], xy[1], angle, mirror_x)
                sp_transform[3] %= 2
            else:
                sp_transform = False

            if subpattern.pattern is not None:
                result = subpattern.pattern.dfs(visit_before=visit_before,
                                                visit_after=visit_after,
                                                transform=sp_transform,
                                                memo=memo,
                                                hierarchy=hierarchy + (self,))
                if result is not subpattern.pattern:
                    # skip assignment to avoid PatternLockedError unless modified
                    subpattern.pattern = result

        if visit_after is not None:
            pat = visit_after(pat, hierarchy=hierarchy, memo=memo, transform=transform)         # type: ignore
        return pat

    def polygonize(self,
                   poly_num_points: Optional[int] = None,
                   poly_max_arclen: Optional[float] = None,
                   ) -> 'Pattern':
        """
        Calls `.to_polygons(...)` on all the shapes in this Pattern and any referenced patterns,
         replacing them with the returned polygons.
        Arguments are passed directly to `shape.to_polygons(...)`.

        Args:
            poly_num_points: Number of points to use for each polygon. Can be overridden by
                `poly_max_arclen` if that results in more points. Optional, defaults to shapes'
                internal defaults.
            poly_max_arclen: Maximum arclength which can be approximated by a single line
             segment. Optional, defaults to shapes' internal defaults.

        Returns:
            self
        """
        old_shapes = self.shapes
        self.shapes = list(chain.from_iterable(
                        (shape.to_polygons(poly_num_points, poly_max_arclen)
                         for shape in old_shapes)))
        for subpat in self.subpatterns:
            if subpat.pattern is not None:
                subpat.pattern.polygonize(poly_num_points, poly_max_arclen)
        return self

    def manhattanize(self,
                     grid_x: numpy.ndarray,
                     grid_y: numpy.ndarray,
                     ) -> 'Pattern':
        """
        Calls `.polygonize()` and `.flatten()` on the pattern, then calls `.manhattanize()` on all the
         resulting shapes, replacing them with the returned Manhattan polygons.

        Args:
            grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
            grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.

        Returns:
            self
        """

        self.polygonize().flatten()
        old_shapes = self.shapes
        self.shapes = list(chain.from_iterable(
                        (shape.manhattanize(grid_x, grid_y) for shape in old_shapes)))
        return self

    def subpatternize(self,
                      recursive: bool = True,
                      norm_value: int = int(1e6),
                      exclude_types: Tuple[Type] = (Polygon,)
                      ) -> 'Pattern':
        """
        Iterates through this `Pattern` and all referenced `Pattern`s. Within each `Pattern`, it iterates
         over all shapes, calling `.normalized_form(norm_value)` on them to retrieve a scale-,
         offset-, dose-, and rotation-independent form. Each shape whose normalized form appears
         more than once is removed and re-added using subpattern objects referencing a newly-created
         `Pattern` containing only the normalized form of the shape.

        Note:
            The default norm_value was chosen to give a reasonable precision when converting
            to GDSII, which uses integer values for pixel coordinates.

        Args:
            recursive: Whether to call recursively on self's subpatterns. Default `True`.
            norm_value: Passed to `shape.normalized_form(norm_value)`. Default `1e6` (see function
                note about GDSII)
            exclude_types: Shape types passed in this argument are always left untouched, for
                speed or convenience. Default: `(shapes.Polygon,)`

        Returns:
            self
        """

        if exclude_types is None:
            exclude_types = ()

        if recursive:
            for subpat in self.subpatterns:
                if subpat.pattern is None:
                    continue
                subpat.pattern.subpatternize(recursive=True,
                                             norm_value=norm_value,
                                             exclude_types=exclude_types)

        # Create a dict which uses the label tuple from `.normalized_form()` as a key, and which
        #  stores `(function_to_create_normalized_shape, [(index_in_shapes, values), ...])`, where
        #  values are the `(offset, scale, rotation, dose)` values as calculated by `.normalized_form()`
        shape_table: MutableMapping[Tuple, List] = defaultdict(lambda: [None, list()])
        for i, shape in enumerate(self.shapes):
            if not any((isinstance(shape, t) for t in exclude_types)):
                label, values, func = shape.normalized_form(norm_value)
                shape_table[label][0] = func
                shape_table[label][1].append((i, values))

        # Iterate over the normalized shapes in the table. If any normalized shape occurs more than
        #  once, create a `Pattern` holding a normalized shape object, and add `self.subpatterns`
        #  entries for each occurrence in self. Also, note down that we should delete the
        #  `self.shapes` entries for which we made SubPatterns.
        shapes_to_remove = []
        for label in shape_table:
            if len(shape_table[label][1]) > 1:
                shape = shape_table[label][0]()
                pat = Pattern(shapes=[shape])

                for i, values in shape_table[label][1]:
                    (offset, scale, rotation, mirror_x, dose) = values
                    subpat = SubPattern(pattern=pat, offset=offset, scale=scale,
                                        rotation=rotation, dose=dose, mirrored=(mirror_x, False))
                    self.subpatterns.append(subpat)
                    shapes_to_remove.append(i)

        # Remove any shapes for which we have created subpatterns.
        for i in sorted(shapes_to_remove, reverse=True):
            del self.shapes[i]

        return self

    def as_polygons(self) -> List[numpy.ndarray]:
        """
        Represents the pattern as a list of polygons.

        Deep-copies the pattern, then calls `.polygonize()` and `.flatten()` on the copy in order to
         generate the list of polygons.

        Returns:
            A list of `(Ni, 2)` `numpy.ndarray`s specifying vertices of the polygons. Each ndarray
             is of the form `[[x0, y0], [x1, y1],...]`.
        """
        pat = self.deepcopy().deepunlock().polygonize().flatten()
        return [shape.vertices + shape.offset for shape in pat.shapes]      # type: ignore      # mypy can't figure out that shapes are all Polygons now

    @overload
    def referenced_patterns_by_id(self) -> Dict[int, 'Pattern']:
        pass

    @overload
    def referenced_patterns_by_id(self, include_none: bool) -> Dict[int, Optional['Pattern']]:
        pass

    def referenced_patterns_by_id(self,
                                  include_none: bool = False,
                                  recursive: bool = True,
                                  ) -> Union[Dict[int, Optional['Pattern']],
                                             Dict[int, 'Pattern']]:

        """
        Create a dictionary with `{id(pat): pat}` for all Pattern objects referenced by this
         Pattern (by default, operates recursively on all referenced Patterns as well).

        Args:
            include_none: If `True`, references to `None` will be included. Default `False`.
            recursive: If `True`, operates recursively on all referenced patterns. Default `True`.

        Returns:
            Dictionary with `{id(pat): pat}` for all referenced Pattern objects
        """
        ids: Dict[int, Optional['Pattern']] = {}
        for subpat in self.subpatterns:
            pat = subpat.pattern
            if id(pat) in ids:
                continue
            if include_none or pat is not None:
                ids[id(pat)] = pat
            if recursive and pat is not None:
                ids.update(pat.referenced_patterns_by_id())
        return ids

    def referenced_patterns_by_name(self, **kwargs) -> List[Tuple[Optional[str], Optional['Pattern']]]:
        """
        Create a list of `(pat.name, pat)` tuples for all Pattern objects referenced by this
         Pattern (operates recursively on all referenced Patterns as well).

        Note that names are not necessarily unique, so a list of tuples is returned
         rather than a dict.

        Args:
            **kwargs: passed to `referenced_patterns_by_id()`.

        Returns:
            List of `(pat.name, pat)` tuples for all referenced Pattern objects
        """
        pats_by_id = self.referenced_patterns_by_id(**kwargs)
        pat_list = [(p.name if p is not None else None, p) for p in pats_by_id.values()]
        return pat_list

    def subpatterns_by_id(self,
                          include_none: bool = False,
                          recursive: bool = True,
                          ) -> Dict[int, List[SubPattern]]:
        """
        Create a dictionary which maps `{id(referenced_pattern): [subpattern0, ...]}`
         for all SubPattern objects referenced by this Pattern (by default, operates
         recursively on all referenced Patterns as well).

        Args:
            include_none: If `True`, references to `None` will be included. Default `False`.
            recursive: If `True`, operates recursively on all referenced patterns. Default `True`.

        Returns:
            Dictionary mapping each pattern id to a list of subpattern objects referencing the pattern.
        """
        ids: Dict[int, List[SubPattern]] = defaultdict(list)
        for subpat in self.subpatterns:
            pat = subpat.pattern
            if include_none or pat is not None:
                ids[id(pat)].append(subpat)
            if recursive and pat is not None:
                ids.update(pat.subpatterns_by_id(include_none=include_none))
        return dict(ids)


    def get_bounds(self) -> Union[numpy.ndarray, None]:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the Pattern's contents in each dimension.
        Returns `None` if the Pattern is empty.

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        if self.is_empty():
            return None

        min_bounds = numpy.array((+inf, +inf))
        max_bounds = numpy.array((-inf, -inf))
        for entry in chain(self.shapes, self.subpatterns, self.labels):
            bounds = entry.get_bounds()
            if bounds is None:
                continue
            min_bounds = numpy.minimum(min_bounds, bounds[0, :])
            max_bounds = numpy.maximum(max_bounds, bounds[1, :])
        if (max_bounds < min_bounds).any():
            return None
        else:
            return numpy.vstack((min_bounds, max_bounds))

    def flatten(self) -> 'Pattern':
        """
        Removes all subpatterns and adds equivalent shapes.

        Shape identifiers are changed to represent their original position in the
         pattern hierarchy:
         `(L1_name (str), L1_index (int), L2_name, L2_index, ..., *original_shape_identifier)`
         where
         `L1_name` is the first-level subpattern's name (e.g. `self.subpatterns[0].pattern.name`),
         `L2_name` is the next-level subpattern's name (e.g.
            `self.subpatterns[0].pattern.subpatterns[0].pattern.name`) and
         `L1_index` is an integer used to differentiate between multiple instance ofi the same
            (or same-named) subpatterns.

        Returns:
            self
        """
        subpatterns = copy.deepcopy(self.subpatterns)
        self.subpatterns = []
        shape_counts: Dict[Tuple, int] = {}
        for subpat in subpatterns:
            if subpat.pattern is None:
                continue
            subpat.pattern.flatten()
            p = subpat.as_pattern()

            # Update identifiers so each shape has a unique one
            for shape in p.shapes:
                combined_identifier = (subpat.pattern.name,) + shape.identifier
                shape_count = shape_counts.get(combined_identifier, 0)
                shape.identifier = (subpat.pattern.name, shape_count) + shape.identifier
                shape_counts[combined_identifier] = shape_count + 1

            self.append(p)
        return self

    def wrap_repeated_shapes(self,
                             name_func: Callable[['Pattern', Union[Shape, Label]], str] = lambda p, s: '_repetition',
                             recursive: bool = True,
                             ) -> 'Pattern':
        """
        Wraps all shapes and labels with a non-`None` `repetition` attribute
          into a `SubPattern`/`Pattern` combination, and applies the `repetition`
          to each `SubPattern` instead of its contained shape.

        Args:
            name_func: Function f(this_pattern, shape) which generates a name for the
                        wrapping pattern. Default always returns '_repetition'.
            recursive: If `True`, this function is also applied to all referenced patterns
                        recursively. Default `True`.

        Returns:
            self
        """
        def do_wrap(pat: Optional[Pattern]) -> Optional[Pattern]:
            if pat is None:
                return pat

            new_subpatterns = []
            for shape in pat.shapes:
                if shape.repetition is None:
                    continue
                new_subpatterns.append(SubPattern(Pattern(name_func(pat, shape), shapes=[shape])))
                shape.repetition = None

            for label in self.labels:
                if label.repetition is None:
                    continue
                new_subpatterns.append(SubPattern(Pattern(name_func(pat, shape), labels=[label])))
                label.repetition = None

            pat.subpatterns += new_subpatterns
            return pat

        if recursive:
            self.apply(do_wrap)
        else:
            do_wrap(self)

        return self


    def translate_elements(self, offset: vector2) -> 'Pattern':
        """
        Translates all shapes, label, and subpatterns by the given offset.

        Args:
            offset: (x, y) to translate by

        Returns:
            self
        """
        for entry in chain(self.shapes, self.subpatterns, self.labels):
            entry.translate(offset)
        return self

    def scale_elements(self, c: float) -> 'Pattern':
        """"
        Scales all shapes and subpatterns by the given value.

        Args:
            c: factor to scale by

        Returns:
            self
        """
        for entry in chain(self.shapes, self.subpatterns):
            entry.scale_by(c)
        return self

    def scale_by(self, c: float) -> 'Pattern':
        """
        Scale this Pattern by the given value
         (all shapes and subpatterns and their offsets are scaled)

        Args:
            c: factor to scale by

        Returns:
            self
        """
        entry: Scalable
        for entry in chain(self.shapes, self.subpatterns):
            entry.offset *= c
            entry.scale_by(c)
        for label in self.labels:
            label.offset *= c
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'Pattern':
        """
        Rotate the Pattern around the a location.

        Args:
            pivot: (x, y) location to rotate around
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        pivot = numpy.array(pivot)
        self.translate_elements(-pivot)
        self.rotate_elements(rotation)
        self.rotate_element_centers(rotation)
        self.translate_elements(+pivot)
        return self

    def rotate_element_centers(self, rotation: float) -> 'Pattern':
        """
        Rotate the offsets of all shapes, labels, and subpatterns around (0, 0)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.subpatterns, self.labels):
            entry.offset = numpy.dot(rotation_matrix_2d(rotation), entry.offset)
        return self

    def rotate_elements(self, rotation: float) -> 'Pattern':
        """
        Rotate each shape and subpattern around its center (offset)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.subpatterns):
            entry.rotate(rotation)
        return self

    def mirror_element_centers(self, axis: int) -> 'Pattern':
        """
        Mirror the offsets of all shapes, labels, and subpatterns across an axis

        Args:
            axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.subpatterns, self.labels):
            entry.offset[axis - 1] *= -1
        return self

    def mirror_elements(self, axis: int) -> 'Pattern':
        """
        Mirror each shape and subpattern across an axis, relative to its
          offset

        Args:
            axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.subpatterns):
            entry.mirror(axis)
        return self

    def mirror(self, axis: int) -> 'Pattern':
        """
        Mirror the Pattern across an axis

        Args:
            axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        self.mirror_elements(axis)
        self.mirror_element_centers(axis)
        return self

    def scale_element_doses(self, c: float) -> 'Pattern':
        """
        Multiply all shape and subpattern doses by a factor

        Args:
            c: Factor to multiply doses by

        Return:
            self
        """
        for entry in chain(self.shapes, self.subpatterns):
            entry.dose *= c
        return self

    def copy(self) -> 'Pattern':
        """
        Return a copy of the Pattern, deep-copying shapes and copying subpattern
         entries, but not deep-copying any referenced patterns.

        See also: `Pattern.deepcopy()`

        Returns:
            A copy of the current Pattern.
        """
        return copy.copy(self)

    def deepcopy(self) -> 'Pattern':
        """
        Convenience method for `copy.deepcopy(pattern)`

        Returns:
            A deep copy of the current Pattern.
        """
        return copy.deepcopy(self)

    def is_empty(self) -> bool:
        """
        Returns:
            True if the pattern is contains no shapes, labels, or subpatterns.
        """
        return (len(self.subpatterns) == 0 and
                len(self.shapes) == 0 and
                len(self.labels) == 0)

    def lock(self) -> 'Pattern':
        """
        Lock the pattern, raising an exception if it is modified.
        Also see `deeplock()`.

        Returns:
            self
        """
        if not self.locked:
            self.shapes = tuple(self.shapes)
            self.labels = tuple(self.labels)
            self.subpatterns = tuple(self.subpatterns)
            LockableImpl.lock(self)
        return self

    def unlock(self) -> 'Pattern':
        """
        Unlock the pattern

        Returns:
            self
        """
        if self.locked:
            LockableImpl.unlock(self)
            self.shapes = list(self.shapes)
            self.labels = list(self.labels)
            self.subpatterns = list(self.subpatterns)
        return self

    def deeplock(self) -> 'Pattern':
        """
        Recursively lock the pattern, all referenced shapes, subpatterns, and labels.

        Returns:
            self
        """
        self.lock()
        for ss in chain(self.shapes, self.labels):
            ss.lock()
        for sp in self.subpatterns:
            sp.deeplock()
        return self

    def deepunlock(self) -> 'Pattern':
        """
        Recursively unlock the pattern, all referenced shapes, subpatterns, and labels.

        This is dangerous unless you have just performed a deepcopy, since anything
          you change will be changed everywhere it is referenced!

        Return:
            self
        """
        self.unlock()
        for ss in chain(self.shapes, self.labels):
            ss.unlock()
        for sp in self.subpatterns:
            sp.deepunlock()
        return self

    @staticmethod
    def load(filename: str) -> 'Pattern':
        """
        Load a Pattern from a file using pickle

        Args:
            filename: Filename to load from

        Returns:
            Loaded Pattern
        """
        with open(filename, 'rb') as f:
            pattern = pickle.load(f)

        return pattern

    def save(self, filename: str) -> 'Pattern':
        """
        Save the Pattern to a file using pickle

        Args:
            filename: Filename to save to

        Returns:
            self
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def visualize(self,
                  offset: vector2 = (0., 0.),
                  line_color: str = 'k',
                  fill_color: str = 'none',
                  overdraw: bool = False):
        """
        Draw a picture of the Pattern and wait for the user to inspect it

        Imports `matplotlib`.

        Note that this can be slow; it is often faster to export to GDSII and use
         klayout or a different GDS viewer!

        Args:
            offset: Coordinates to offset by before drawing
            line_color: Outlines are drawn with this color (passed to `matplotlib.collections.PolyCollection`)
            fill_color: Interiors are drawn with this color (passed to `matplotlib.collections.PolyCollection`)
            overdraw: Whether to create a new figure or draw on a pre-existing one
        """
        # TODO: add text labels to visualize()
        from matplotlib import pyplot       # type: ignore
        import matplotlib.collections       # type: ignore

        offset = numpy.array(offset, dtype=float)

        if not overdraw:
            figure = pyplot.figure()
            pyplot.axis('equal')
        else:
            figure = pyplot.gcf()

        axes = figure.gca()

        polygons = []
        for shape in self.shapes:
            polygons += [offset + s.offset + s.vertices for s in shape.to_polygons()]

        mpl_poly_collection = matplotlib.collections.PolyCollection(polygons,
                                                                    facecolors=fill_color,
                                                                    edgecolors=line_color)
        axes.add_collection(mpl_poly_collection)
        pyplot.axis('equal')

        for subpat in self.subpatterns:
            subpat.as_pattern().visualize(offset=offset, overdraw=True,
                                          line_color=line_color, fill_color=fill_color)

        if not overdraw:
            pyplot.show()

    @staticmethod
    def find_toplevel(patterns: Iterable['Pattern']) -> List['Pattern']:
        """
        Given a list of Pattern objects, return those that are not referenced by
         any other pattern.

        Args:
            patterns: A list of patterns to filter.

        Returns:
            A filtered list in which no pattern is referenced by any other pattern.
        """
        def get_children(pat: Pattern, memo: Set) -> Set:
            children = set(sp.pattern for sp in pat.subpatterns if sp.pattern is not None)
            new_children = children - memo
            memo |= new_children

            for child_pat in new_children:
                memo |= get_children(child_pat, memo)
            return memo

        patterns = set(patterns)
        not_toplevel: Set['Pattern'] = set()
        for pattern in patterns:
            not_toplevel |= get_children(pattern, not_toplevel)

        toplevel = list(patterns - not_toplevel)
        return toplevel

    def __repr__(self) -> str:
        locked = ' L' if self.locked else ''
        return (f'<Pattern "{self.name}": sh{len(self.shapes)} sp{len(self.subpatterns)} la{len(self.labels)}{locked}>')
