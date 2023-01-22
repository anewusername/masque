"""
 Base object representing a lithography mask.
"""

from typing import List, Callable, Tuple, Dict, Union, Set, Sequence, Optional, Type, overload, cast
from typing import Mapping, MutableMapping, Iterable, TypeVar, Any
import copy
from itertools import chain
from collections import defaultdict

import numpy
from numpy import inf
from numpy.typing import NDArray, ArrayLike
# .visualize imports matplotlib and matplotlib.collections

from .refs import Ref
from .shapes import Shape, Polygon
from .label import Label
from .utils import rotation_matrix_2d, normalize_mirror, AutoSlots, annotations_t
from .error import PatternError
from .traits import AnnotatableImpl, Scalable, Mirrorable, Rotatable, Positionable, Repeatable
from .ports import Port, PortList


P = TypeVar('P', bound='Pattern')


class Pattern(PortList, AnnotatableImpl, Mirrorable, metaclass=AutoSlots):
    """
    2D layout consisting of some set of shapes, labels, and references to other Pattern objects
     (via Ref). Shapes are assumed to inherit from masque.shapes.Shape or provide equivalent functions.
    """
    __slots__ = ('shapes', 'labels', 'refs', 'ports')

    shapes: List[Shape]
    """ List of all shapes in this Pattern.
    Elements in this list are assumed to inherit from Shape or provide equivalent functions.
    """

    labels: List[Label]
    """ List of all labels in this Pattern. """

    refs: List[Ref]
    """ List of all references to other patterns (`Ref`s) in this `Pattern`.
    Multiple objects in this list may reference the same Pattern object
      (i.e. multiple instances of the same object).
    """

    ports: Dict[str, Port]
    """ Uniquely-named ports which can be used to snap to other Pattern instances"""

    def __init__(
            self,
            *,
            shapes: Sequence[Shape] = (),
            labels: Sequence[Label] = (),
            refs: Sequence[Ref] = (),
            annotations: Optional[annotations_t] = None,
            ports: Optional[Mapping[str, Port]] = None
            ) -> None:
        """
        Basic init; arguments get assigned to member variables.
         Non-list inputs for shapes and refs get converted to lists.

        Args:
            shapes: Initial shapes in the Pattern
            labels: Initial labels in the Pattern
            refs: Initial refs in the Pattern
            annotations: Initial annotations for the pattern
            ports: Any ports in the pattern
        """
        if isinstance(shapes, list):
            self.shapes = shapes
        else:
            self.shapes = list(shapes)

        if isinstance(labels, list):
            self.labels = labels
        else:
            self.labels = list(labels)

        if isinstance(refs, list):
            self.refs = refs
        else:
            self.refs = list(refs)

        if ports is not None:
            ports = dict(copy.deepcopy(ports))

        self.annotations = annotations if annotations is not None else {}

    def __repr__(self) -> str:
        s = f'<Pattern: sh{len(self.shapes)} sp{len(self.refs)} la{len(self.labels)} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s

    def __copy__(self) -> 'Pattern':
        return Pattern(
            shapes=copy.deepcopy(self.shapes),
            labels=copy.deepcopy(self.labels),
            refs=[copy.copy(sp) for sp in self.refs],
            annotations=copy.deepcopy(self.annotations),
            ports=copy.deepcopy(self.ports),
            )

    def __deepcopy__(self, memo: Optional[Dict] = None) -> 'Pattern':
        memo = {} if memo is None else memo
        new = Pattern(
            shapes=copy.deepcopy(self.shapes, memo),
            labels=copy.deepcopy(self.labels, memo),
            refs=copy.deepcopy(self.refs, memo),
            annotations=copy.deepcopy(self.annotations, memo),
            ports=copy.deepcopy(self.ports),
            )
        return new

    def append(self: P, other_pattern: Pattern) -> P:
        """
        Appends all shapes, labels and refs from other_pattern to self's shapes,
          labels, and supbatterns.

        Args:
           other_pattern: The Pattern to append

        Returns:
            self
        """
        self.refs += other_pattern.refs
        self.shapes += other_pattern.shapes
        self.labels += other_pattern.labels
        self.annotations += other_pattern.annotations
        self.ports += other_pattern.ports
        return self

    def subset(
            self,
            shapes: Optional[Callable[[Shape], bool]] = None,
            labels: Optional[Callable[[Label], bool]] = None,
            refs: Optional[Callable[[Ref], bool]] = None,
            annotations: Optional[Callable[[annotation_t], bool]] = None,
            ports: Optional[Callable[[str], bool]] = None,
            default_keep: bool = False
            ) -> 'Pattern':
        """
        Returns a Pattern containing only the entities (e.g. shapes) for which the
          given entity_func returns True.
        Self is _not_ altered, but shapes, labels, and refs are _not_ copied, just referenced.

        Args:
            shapes: Given a shape, returns a boolean denoting whether the shape is a member of the subset.
            labels: Given a label, returns a boolean denoting whether the label is a member of the subset.
            refs: Given a ref, returns a boolean denoting if it is a member of the subset.
            annotations: Given an annotation, returns a boolean denoting if it is a member of the subset.
            ports: Given a port, returns a boolean denoting if it is a member of the subset.
            default_keep: If `True`, keeps all elements of a given type if no function is supplied.
                Default `False` (discards all elements).

        Returns:
            A Pattern containing all the shapes and refs for which the parameter
                functions return True
        """
        pat = Pattern()

        if shapes is not None:
            pat.shapes = [s for s in self.shapes if shapes(s)]
        elif default_keep:
            pat.shapes = copy.copy(self.shapes)

        if labels is not None:
            pat.labels = [s for s in self.labels if labels(s)]
        elif default_keep:
            pat.labels = copy.copy(self.labels)

        if refs is not None:
            pat.refs = [s for s in self.refs if refs(s)]
        elif default_keep:
            pat.refs = copy.copy(self.refs)

        if annotations is not None:
            pat.annotations = [s for s in self.annotations if annotations(s)]
        elif default_keep:
            pat.annotations = copy.copy(self.annotations)

        if ports is not None:
            pat.ports = {k: v for k, v in self.ports.items() if ports(k)}
        elif default_keep:
            pat.ports = copy.copy(self.ports)

        return pat

    def polygonize(
            self: P,
            poly_num_points: Optional[int] = None,
            poly_max_arclen: Optional[float] = None,
            ) -> P:
        """
        Calls `.to_polygons(...)` on all the shapes in this Pattern, replacing them with the returned polygons.
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
        self.shapes = list(chain.from_iterable((
            shape.to_polygons(poly_num_points, poly_max_arclen)
            for shape in old_shapes)))
        return self

    def manhattanize(
            self: P,
            grid_x: ArrayLike,
            grid_y: ArrayLike,
            ) -> P:
        """
        Calls `.polygonize()` on the pattern, then calls `.manhattanize()` on all the
         resulting shapes, replacing them with the returned Manhattan polygons.

        Args:
            grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
            grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.

        Returns:
            self
        """

        self.polygonize()
        old_shapes = self.shapes
        self.shapes = list(chain.from_iterable(
            (shape.manhattanize(grid_x, grid_y) for shape in old_shapes)))
        return self

    def as_polygons(self, library: Mapping[str, 'Pattern']) -> List[NDArray[numpy.float64]]:
        """
        Represents the pattern as a list of polygons.

        Deep-copies the pattern, then calls `.polygonize()` and `.flatten()` on the copy in order to
         generate the list of polygons.

        Returns:
            A list of `(Ni, 2)` `numpy.ndarray`s specifying vertices of the polygons. Each ndarray
             is of the form `[[x0, y0], [x1, y1],...]`.
        """
        pat = self.deepcopy().polygonize().flatten(library=library)
        return [shape.vertices + shape.offset for shape in pat.shapes]      # type: ignore      # mypy can't figure out that shapes are all Polygons now

    def referenced_patterns(self) -> Set[Optional[str]]:
        """
        Get all pattern namers referenced by this pattern. Non-recursive.

        Returns:
            A set of all pattern names referenced by this pattern.
        """
        return set(sp.target for sp in self.refs)

    def get_bounds(
            self,
            library: Optional[Mapping[str, 'Pattern']] = None,
            ) -> Optional[NDArray[numpy.float64]]:
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

        for entry in chain(self.shapes, self.labels):
            bounds = entry.get_bounds()
            if bounds is None:
                continue
            min_bounds = numpy.minimum(min_bounds, bounds[0, :])
            max_bounds = numpy.maximum(max_bounds, bounds[1, :])

        if self.refs and (library is None):
            raise PatternError('Must provide a library to get_bounds() to resolve refs')

        for entry in self.refs:
            bounds = entry.get_bounds(library=library)
            if bounds is None:
                continue
            min_bounds = numpy.minimum(min_bounds, bounds[0, :])
            max_bounds = numpy.maximum(max_bounds, bounds[1, :])

        if (max_bounds < min_bounds).any():
            return None
        else:
            return numpy.vstack((min_bounds, max_bounds))

    def get_bounds_nonempty(
            self,
            library: Optional[Mapping[str, 'Pattern']] = None,
            ) -> NDArray[numpy.float64]:
        """
        Convenience wrapper for `get_bounds()` which asserts that the Pattern as non-None bounds.

        Returns:
            `[[x_min, y_min], [x_max, y_max]]`
        """
        bounds = self.get_bounds(library)
        assert(bounds is not None)
        return bounds

    def translate_elements(self: P, offset: ArrayLike) -> P:
        """
        Translates all shapes, label, and refs by the given offset.

        Args:
            offset: (x, y) to translate by

        Returns:
            self
        """
        for entry in chain(self.shapes, self.refs, self.labels, self.ports):
            cast(Positionable, entry).translate(offset)
        return self

    def scale_elements(self: P, c: float) -> P:
        """"
        Scales all shapes and refs by the given value.

        Args:
            c: factor to scale by

        Returns:
            self
        """
        for entry in chain(self.shapes, self.refs):
            cast(Scalable, entry).scale_by(c)
        return self

    def scale_by(self: P, c: float) -> P:
        """
        Scale this Pattern by the given value
         (all shapes and refs and their offsets are scaled)

        Args:
            c: factor to scale by

        Returns:
            self
        """
        for entry in chain(self.shapes, self.refs):
            cast(Positionable, entry).offset *= c
            cast(Scalable, entry).scale_by(c)

            rep = cast(Repeatable, entry).repetition
            if rep:
                rep.scale_by(c)

        for label in self.labels:
            cast(Positionable, label).offset *= c

            rep = cast(Repeatable, label).repetition
            if rep:
                rep.scale_by(c)

        for port in self.ports.values():
            port.offset *= c
        return self

    def rotate_around(self: P, pivot: ArrayLike, rotation: float) -> P:
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

    def rotate_element_centers(self: P, rotation: float) -> P:
        """
        Rotate the offsets of all shapes, labels, and refs around (0, 0)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.refs, self.labels, self.ports):
            old_offset = cast(Positionable, entry).offset
            cast(Positionable, entry).offset = numpy.dot(rotation_matrix_2d(rotation), old_offset)
        return self

    def rotate_elements(self: P, rotation: float) -> P:
        """
        Rotate each shape and refs around its center (offset)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.refs):
            cast(Rotatable, entry).rotate(rotation)
        return self

    def mirror_element_centers(self: P, axis: int) -> P:
        """
        Mirror the offsets of all shapes, labels, and refs across an axis

        Args:
            axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.refs, self.labels, self.ports):
            cast(Positionable, entry).offset[axis - 1] *= -1
        return self

    def mirror_elements(self: P, axis: int) -> P:
        """
        Mirror each shape and refs across an axis, relative to its
          offset

        Args:
            axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for entry in chain(self.shapes, self.refs):
            cast(Mirrorable, entry).mirror(axis)
        return self

    def mirror(self: P, axis: int) -> P:
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

    def copy(self: P) -> P:
        """
        Return a copy of the Pattern, deep-copying shapes and copying refs
         entries, but not deep-copying any referenced patterns.

        See also: `Pattern.deepcopy()`

        Returns:
            A copy of the current Pattern.
        """
        return copy.copy(self)

    def deepcopy(self: P) -> P:
        """
        Convenience method for `copy.deepcopy(pattern)`

        Returns:
            A deep copy of the current Pattern.
        """
        return copy.deepcopy(self)

    def is_empty(self) -> bool:
        """
        Returns:
            True if the pattern is contains no shapes, labels, or refs.
        """
        return (len(self.refs) == 0
                and len(self.shapes) == 0
                and len(self.labels) == 0)

    def ref(self: P, *args: Any, **kwargs: Any) -> P:
        """
        Convenience function which constructs a `Ref` object and adds it
         to this pattern.

        Args:
            *args: Passed to `Ref()`
            **kwargs: Passed to `Ref()`

        Returns:
            self
        """
        self.refs.append(Ref(*args, **kwargs))
        return self

    def flatten(
            self: P,
            library: Mapping[str, P],
            ) -> 'Pattern':
        """
        Removes all refs (recursively) and adds equivalent shapes.
        Alters the current pattern in-place

        Args:
            library: Source for referenced patterns.

        Returns:
            self
        """
        flattened: Dict[Optional[str], Optional[P]] = {}

        def flatten_single(name: Optional[str]) -> None:
            if name is None:
                pat = self
            else:
                pat = library[name].deepcopy()
                flattened[name] = None

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

        flatten_single(None)
        return self

    def visualize(
            self: P,
            library: Optional[Mapping[str, P]] = None,
            offset: ArrayLike = (0., 0.),
            line_color: str = 'k',
            fill_color: str = 'none',
            overdraw: bool = False,
            ) -> None:
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

        if self.refs and library is None:
            raise PatternError('Must provide a library when visualizing a pattern with refs')

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

        mpl_poly_collection = matplotlib.collections.PolyCollection(
            polygons,
            facecolors=fill_color,
            edgecolors=line_color,
            )
        axes.add_collection(mpl_poly_collection)
        pyplot.axis('equal')

        for ref in self.refs:
            ref.as_pattern(library=library).visualize(
                library=library,
                offset=offset,
                overdraw=True,
                line_color=line_color,
                fill_color=fill_color,
                )

        if not overdraw:
            pyplot.xlabel('x')
            pyplot.ylabel('y')
            pyplot.show()
