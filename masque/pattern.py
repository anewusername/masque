"""
 Base object representing a lithography mask.
"""
from typing import Callable, Sequence, cast, Mapping, Self, Any, Iterable, TypeVar, MutableMapping
import copy
import logging
from itertools import chain
from collections import defaultdict

import numpy
from numpy import inf, pi
from numpy.typing import NDArray, ArrayLike
# .visualize imports matplotlib and matplotlib.collections

from .ref import Ref
from .shapes import Shape, Polygon, Path, DEFAULT_POLY_NUM_VERTICES
from .label import Label
from .utils import rotation_matrix_2d, annotations_t, layer_t, normalize_mirror
from .error import PatternError
from .traits import AnnotatableImpl, Scalable, Mirrorable, Rotatable, Positionable, Repeatable, Bounded
from .ports import Port, PortList


logger = logging.getLogger(__name__)


class Pattern(PortList, AnnotatableImpl, Mirrorable):
    """
    2D layout consisting of some set of shapes, labels, and references to other Pattern objects
     (via Ref). Shapes are assumed to inherit from masque.shapes.Shape or provide equivalent functions.
    """
    __slots__ = (
        'shapes', 'labels', 'refs', '_ports',
        # inherited
        '_offset', '_annotations',
        )

    shapes: defaultdict[layer_t, list[Shape]]
    """ Stores of all shapes in this Pattern, indexed by layer.
    Elements in this list are assumed to inherit from Shape or provide equivalent functions.
    """

    labels: defaultdict[layer_t, list[Label]]
    """ List of all labels in this Pattern. """

    refs: defaultdict[str | None, list[Ref]]
    """ List of all references to other patterns (`Ref`s) in this `Pattern`.
    Multiple objects in this list may reference the same Pattern object
      (i.e. multiple instances of the same object).
    """

    _ports: dict[str, Port]
    """ Uniquely-named ports which can be used to snap to other Pattern instances"""

    @property
    def ports(self) -> dict[str, Port]:
        return self._ports

    @ports.setter
    def ports(self, value: dict[str, Port]) -> None:
        self._ports = value

    def __init__(
            self,
            *,
            shapes: Mapping[layer_t, Sequence[Shape]] | None = None,
            labels: Mapping[layer_t, Sequence[Label]] | None = None,
            refs: Mapping[str | None, Sequence[Ref]] | None = None,
            annotations: annotations_t | None = None,
            ports: Mapping[str, 'Port'] | None = None
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
        self.shapes = defaultdict(list)
        self.labels = defaultdict(list)
        self.refs = defaultdict(list)
        if shapes:
            for layer, sseq in shapes.items():
                self.shapes[layer].extend(sseq)
        if labels:
            for layer, lseq in labels.items():
                self.labels[layer].extend(lseq)
        if refs:
            for target, rseq in refs.items():
                self.refs[target].extend(rseq)

        if ports is not None:
            self.ports = dict(copy.deepcopy(ports))
        else:
            self.ports = {}

        self.annotations = annotations if annotations is not None else {}

    def __repr__(self) -> str:
        nshapes = sum(len(seq) for seq in self.shapes.values())
        nrefs = sum(len(seq) for seq in self.refs.values())
        nlabels = sum(len(seq) for seq in self.labels.values())

        s = f'<Pattern: s{nshapes} r{nrefs} l{nlabels} ['
        for name, port in self.ports.items():
            s += f'\n\t{name}: {port}'
        s += ']>'
        return s

    def __copy__(self) -> 'Pattern':
        logger.warning('Making a shallow copy of a Pattern... old shapes are re-referenced!')
        new = Pattern(
            annotations=copy.deepcopy(self.annotations),
            ports=copy.deepcopy(self.ports),
            )
        for target, rseq in self.refs.items():
            new.refs[target].extend(rseq)
        for layer, sseq in self.shapes.items():
            new.shapes[layer].extend(sseq)
        for layer, lseq in self.labels.items():
            new.labels[layer].extend(lseq)

        return new

#    def __deepcopy__(self, memo: dict | None = None) -> 'Pattern':
#        memo = {} if memo is None else memo
#        new = Pattern(
#            shapes=copy.deepcopy(self.shapes, memo),
#            labels=copy.deepcopy(self.labels, memo),
#            refs=copy.deepcopy(self.refs, memo),
#            annotations=copy.deepcopy(self.annotations, memo),
#            ports=copy.deepcopy(self.ports),
#            )
#        return new

    def append(self, other_pattern: 'Pattern') -> Self:
        """
        Appends all shapes, labels and refs from other_pattern to self's shapes,
          labels, and supbatterns.

        Args:
           other_pattern: The Pattern to append

        Returns:
            self
        """
        for target, rseq in other_pattern.refs.items():
            self.refs[target].extend(rseq)
        for layer, sseq in other_pattern.shapes.items():
            self.shapes[layer].extend(sseq)
        for layer, lseq in other_pattern.labels.items():
            self.labels[layer].extend(lseq)

        annotation_conflicts = set(self.annotations.keys()) & set(other_pattern.annotations.keys())
        if annotation_conflicts:
            raise PatternError(f'Annotation keys overlap: {annotation_conflicts}')
        self.annotations.update(other_pattern.annotations)

        port_conflicts = set(self.ports.keys()) & set(other_pattern.ports.keys())
        if port_conflicts:
            raise PatternError(f'Port names overlap: {port_conflicts}')
        self.ports.update(other_pattern.ports)

        return self

    def subset(
            self,
            shapes: Callable[[layer_t, Shape], bool] | None = None,
            labels: Callable[[layer_t, Label], bool] | None = None,
            refs: Callable[[str | None, Ref], bool] | None = None,
            annotations: Callable[[str, list[int | float | str]], bool] | None = None,
            ports: Callable[[str, Port], bool] | None = None,
            default_keep: bool = False
            ) -> 'Pattern':
        """
        Returns a Pattern containing only the entities (e.g. shapes) for which the
          given entity_func returns True.
        Self is _not_ altered, but shapes, labels, and refs are _not_ copied, just referenced.

        Args:
            shapes: Given a layer and shape, returns a boolean denoting whether the shape is a
                member of the subset.
            labels: Given a layer and label, returns a boolean denoting whether the label is a
                member of the subset.
            refs: Given a target and ref, returns a boolean denoting if it is a member of the subset.
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
            for layer in self.shapes:
                pat.shapes[layer] = [ss for ss in self.shapes[layer] if shapes(layer, ss)]
        elif default_keep:
            pat.shapes = copy.copy(self.shapes)

        if labels is not None:
            for layer in self.labels:
                pat.labels[layer] = [ll for ll in self.labels[layer] if labels(layer, ll)]
        elif default_keep:
            pat.labels = copy.copy(self.labels)

        if refs is not None:
            for target in self.refs:
                pat.refs[target] = [rr for rr in self.refs[target] if refs(target, rr)]
        elif default_keep:
            pat.refs = copy.copy(self.refs)

        if annotations is not None:
            pat.annotations = {k: v for k, v in self.annotations.items() if annotations(k, v)}
        elif default_keep:
            pat.annotations = copy.copy(self.annotations)

        if ports is not None:
            pat.ports = {k: v for k, v in self.ports.items() if ports(k, v)}
        elif default_keep:
            pat.ports = copy.copy(self.ports)

        return pat

    def polygonize(
            self,
            num_vertices: int | None = DEFAULT_POLY_NUM_VERTICES,
            max_arclen: float | None = None,
            ) -> Self:
        """
        Calls `.to_polygons(...)` on all the shapes in this Pattern, replacing them with the returned polygons.
        Arguments are passed directly to `shape.to_polygons(...)`.

        Args:
            num_vertices: Number of points to use for each polygon. Can be overridden by
                `max_arclen` if that results in more points. Optional, defaults to shapes'
                internal defaults.
            max_arclen: Maximum arclength which can be approximated by a single line
             segment. Optional, defaults to shapes' internal defaults.

        Returns:
            self
        """
        for layer in self.shapes:
            self.shapes[layer] = list(chain.from_iterable(
                ss.to_polygons(num_vertices, max_arclen)
                for ss in self.shapes[layer]
                ))
        return self

    def manhattanize(
            self,
            grid_x: ArrayLike,
            grid_y: ArrayLike,
            ) -> Self:
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
        for layer in self.shapes:
            self.shapes[layer] = list(chain.from_iterable((
                ss.manhattanize(grid_x, grid_y)
                for ss in self.shapes[layer]
                )))
        return self

    def as_polygons(self, library: Mapping[str, 'Pattern']) -> list[NDArray[numpy.float64]]:
        """
        Represents the pattern as a list of polygons.

        Deep-copies the pattern, then calls `.polygonize()` and `.flatten()` on the copy in order to
         generate the list of polygons.

        Returns:
            A list of `(Ni, 2)` `numpy.ndarray`s specifying vertices of the polygons. Each ndarray
             is of the form `[[x0, y0], [x1, y1],...]`.
        """
        pat = self.deepcopy().polygonize().flatten(library=library)
        polys = [
            cast(Polygon, shape).vertices + cast(Polygon, shape).offset
            for shape in chain_elements(pat.shapes)
            ]
        return polys

    def referenced_patterns(self) -> set[str | None]:
        """
        Get all pattern namers referenced by this pattern. Non-recursive.

        Returns:
            A set of all pattern names referenced by this pattern.
        """
        return set(self.refs.keys())

    def get_bounds(
            self,
            library: Mapping[str, 'Pattern'] | None = None,
            recurse: bool = True,
            cache: MutableMapping[str, NDArray[numpy.float64]] | None = None,
            ) -> NDArray[numpy.float64] | None:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the Pattern's contents in each dimension.
        Returns `None` if the Pattern is empty.

        Args:
            TODO docs for get_bounds

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        if self.is_empty():
            return None

        cbounds = numpy.array([
            (+inf, +inf),
            (-inf, -inf),
            ])

        for entry in chain_elements(self.shapes, self.labels):
            bounds = cast(Bounded, entry).get_bounds()
            if bounds is None:
                continue
            if entry.repetition is not None:
                bounds += entry.repetition.get_bounds()
            cbounds[0] = numpy.minimum(cbounds[0], bounds[0])
            cbounds[1] = numpy.maximum(cbounds[1], bounds[1])

        if recurse and self.has_refs():
            if library is None:
                raise PatternError('Must provide a library to get_bounds() to resolve refs')

            if cache is None:
                cache = {}

            for target, refs in self.refs.items():
                if target is None:
                    continue
                if not refs:
                    continue

                if target in cache:
                    unrot_bounds = cache[target]
                elif any(numpy.isclose(ref.rotation % pi / 2, 0) for ref in refs):
                    unrot_bounds = library[target].get_bounds(library=library, recurse=recurse, cache=cache)
                    cache[target] = unrot_bounds

                for ref in refs:
                    if numpy.isclose(ref.rotation % (pi / 2), 0):
                        if unrot_bounds is None:
                            bounds = None
                        else:
                            ubounds = unrot_bounds.copy()
                            mirr_x, rot2 = normalize_mirror(ref.mirrored)
                            if mirr_x:
                                ubounds[:, 1] *= -1
                            bounds = numpy.round(rotation_matrix(ref.rotation + rot2)) @ ubounds
                            # note: rounding fixes up

                    else:
                        # Non-manhattan rotation, have to figure out bounds by rotating the pattern
                        bounds = ref.get_bounds(library[target], library=library)

                    if bounds is None:
                        continue

                    if ref.repetition is not None:
                        bounds += ref.repetition.get_bounds()

                    cbounds[0] = numpy.minimum(cbounds[0], bounds[0])
                    cbounds[1] = numpy.maximum(cbounds[1], bounds[1])

        if (cbounds[1] < cbounds[0]).any():
            return None
        else:
            return cbounds

    def get_bounds_nonempty(
            self,
            library: Mapping[str, 'Pattern'] | None = None,
            recurse: bool = True,
            ) -> NDArray[numpy.float64]:
        """
        Convenience wrapper for `get_bounds()` which asserts that the Pattern as non-None bounds.

        Args:
            TODO docs for get_bounds

        Returns:
            `[[x_min, y_min], [x_max, y_max]]`
        """
        bounds = self.get_bounds(library)
        assert bounds is not None
        return bounds

    def translate_elements(self, offset: ArrayLike) -> Self:
        """
        Translates all shapes, label, refs, and ports by the given offset.

        Args:
            offset: (x, y) to translate by

        Returns:
            self
        """
        for entry in chain(chain_elements(self.shapes, self.labels, self.refs), self.ports.values()):
            cast(Positionable, entry).translate(offset)
        return self

    def scale_elements(self, c: float) -> Self:
        """"
        Scales all shapes and refs by the given value.

        Args:
            c: factor to scale by

        Returns:
            self
        """
        for entry in chain_elements(self.shapes, self.refs):
            cast(Scalable, entry).scale_by(c)
        return self

    def scale_by(self, c: float) -> Self:
        """
        Scale this Pattern by the given value
         (all shapes and refs and their offsets are scaled,
          as are all label and port offsets)

        Args:
            c: factor to scale by

        Returns:
            self
        """
        for entry in chain_elements(self.shapes, self.refs):
            cast(Positionable, entry).offset *= c
            cast(Scalable, entry).scale_by(c)

            rep = cast(Repeatable, entry).repetition
            if rep:
                rep.scale_by(c)

        for label in chain_elements(self.labels):
            cast(Positionable, label).offset *= c

            rep = cast(Repeatable, label).repetition
            if rep:
                rep.scale_by(c)

        for port in self.ports.values():
            port.offset *= c
        return self

    def rotate_around(self, pivot: ArrayLike, rotation: float) -> Self:
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

    def rotate_element_centers(self, rotation: float) -> Self:
        """
        Rotate the offsets of all shapes, labels, refs, and ports around (0, 0)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for entry in chain(chain_elements(self.shapes, self.refs, self.labels), self.ports.values()):
            old_offset = cast(Positionable, entry).offset
            cast(Positionable, entry).offset = numpy.dot(rotation_matrix_2d(rotation), old_offset)
        return self

    def rotate_elements(self, rotation: float) -> Self:
        """
        Rotate each shape, ref, and port around its origin (offset)

        Args:
            rotation: Angle to rotate by (counter-clockwise, radians)

        Returns:
            self
        """
        for entry in chain(chain_elements(self.shapes, self.refs), self.ports.values()):
            cast(Rotatable, entry).rotate(rotation)
        return self

    def mirror_element_centers(self, across_axis: int) -> Self:
        """
        Mirror the offsets of all shapes, labels, and refs across an axis

        Args:
            across_axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for entry in chain(chain_elements(self.shapes, self.refs, self.labels), self.ports.values()):
            cast(Positionable, entry).offset[across_axis - 1] *= -1
        return self

    def mirror_elements(self, across_axis: int) -> Self:
        """
        Mirror each shape, ref, and pattern across an axis, relative
          to its offset

        Args:
            across_axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        for entry in chain(chain_elements(self.shapes, self.refs), self.ports.values()):
            cast(Mirrorable, entry).mirror(across_axis)
        return self

    def mirror(self, across_axis: int) -> Self:
        """
        Mirror the Pattern across an axis

        Args:
            across_axis: Axis to mirror across
                (0: mirror across x axis, 1: mirror across y axis)

        Returns:
            self
        """
        self.mirror_elements(across_axis)
        self.mirror_element_centers(across_axis)
        return self

    def copy(self) -> Self:
        """
        Return a copy of the Pattern, deep-copying shapes and copying refs
         entries, but not deep-copying any referenced patterns.

        See also: `Pattern.deepcopy()`

        Returns:
            A copy of the current Pattern.
        """
        return copy.copy(self)

    def deepcopy(self) -> Self:
        """
        Convenience method for `copy.deepcopy(pattern)`

        Returns:
            A deep copy of the current Pattern.
        """
        return copy.deepcopy(self)

    def is_empty(self) -> bool:
        """
        # TODO is_empty doesn't include ports... maybe there should be an equivalent?
        Returns:
            True if the pattern is contains no shapes, labels, or refs.
        """
        return not (self.has_refs() or self.has_shapes() or self.has_labels())

    def has_refs(self) -> bool:
        return any(True for _ in chain.from_iterable(self.refs.values()))

    def has_shapes(self) -> bool:
        return any(True for _ in chain.from_iterable(self.shapes.values()))

    def has_labels(self) -> bool:
        return any(True for _ in chain.from_iterable(self.labels.values()))

    def ref(self, target: str | None, *args: Any, **kwargs: Any) -> Self:
        """
        Convenience function which constructs a `Ref` object and adds it
         to this pattern.

        Args:
            target: Target for the ref
            *args: Passed to `Ref()`
            **kwargs: Passed to `Ref()`

        Returns:
            self
        """
        self.refs[target].append(Ref(*args, **kwargs))
        return self

    def polygon(self, layer: layer_t, *args: Any, **kwargs: Any) -> Self:
        """
        Convenience function which constructs a `Polygon` object and adds it
         to this pattern.

        Args:
            layer: Layer for the polygon
            *args: Passed to `Polygon()`
            **kwargs: Passed to `Polygon()`

        Returns:
            self
        """
        self.shapes[layer].append(Polygon(*args, **kwargs))
        return self

    def rect(self, layer: layer_t, *args: Any, **kwargs: Any) -> Self:
        """
        Convenience function which calls `Polygon.rect` to construct a
         rectangle and adds it to this pattern.

        Args:
            layer: Layer for the rectangle
            *args: Passed to `Polygon.rect()`
            **kwargs: Passed to `Polygon.rect()`

        Returns:
            self
        """
        self.shapes[layer].append(Polygon.rect(*args, **kwargs))
        return self

    def path(self, layer: layer_t, *args: Any, **kwargs: Any) -> Self:
        """
        Convenience function which constructs a `Path` object and adds it
         to this pattern.

        Args:
            layer: Layer for the path
            *args: Passed to `Path()`
            **kwargs: Passed to `Path()`

        Returns:
            self
        """
        self.shapes[layer].append(Path(*args, **kwargs))
        return self

    def label(self, layer: layer_t, *args: Any, **kwargs: Any) -> Self:
        """
        Convenience function which constructs a `Label` object
         and adds it to this pattern.

        Args:
            layer: Layer for the label
            *args: Passed to `Label()`
            **kwargs: Passed to `Label()`

        Returns:
            self
        """
        self.labels[layer].append(Label(*args, **kwargs))
        return self

    def flatten(
            self,
            library: Mapping[str, 'Pattern'],
            flatten_ports: bool = False,       # TODO document
            ) -> 'Pattern':
        """
        Removes all refs (recursively) and adds equivalent shapes.
        Alters the current pattern in-place

        Args:
            library: Source for referenced patterns.

        Returns:
            self
        """
        flattened: dict[str | None, 'Pattern | None'] = {}

        # TODO both Library and Pattern have flatten()... pattern is in-place?
        def flatten_single(name: str | None) -> None:
            if name is None:
                pat = self
            else:
                pat = library[name].deepcopy()
                flattened[name] = None

            for target, refs in pat.refs.items():
                if target is None:
                    continue
                if not refs:
                    continue

                if target not in flattened:
                    flatten_single(target)
                target_pat = flattened[target]

                if target_pat is None:
                    raise PatternError(f'Circular reference in {name} to {target}')
                if target_pat.is_empty():        # avoid some extra allocations
                    continue

                for ref in refs:
                    p = ref.as_pattern(pattern=target_pat)
                    if not flatten_ports:
                        p.ports.clear()
                    pat.append(p)

            pat.refs.clear()
            flattened[name] = pat

        flatten_single(None)
        return self

    def visualize(
            self,
            library: Mapping[str, 'Pattern'] | None = None,
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

        if self.has_refs() and library is None:
            raise PatternError('Must provide a library when visualizing a pattern with refs')

        offset = numpy.array(offset, dtype=float)

        if not overdraw:
            figure = pyplot.figure()
            pyplot.axis('equal')
        else:
            figure = pyplot.gcf()

        axes = figure.gca()

        polygons = []
        for shape in chain.from_iterable(self.shapes.values()):
            polygons += [offset + s.offset + s.vertices for s in shape.to_polygons()]

        mpl_poly_collection = matplotlib.collections.PolyCollection(
            polygons,
            facecolors=fill_color,
            edgecolors=line_color,
            )
        axes.add_collection(mpl_poly_collection)
        pyplot.axis('equal')

        for target, refs in self.refs.items():
            if target is None:
                continue
            if not refs:
                continue
            assert library is not None
            target_pat = library[target]
            for ref in refs:
                ref.as_pattern(target_pat).visualize(
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


TT = TypeVar('TT')


def chain_elements(*args: Mapping[Any, Iterable[TT]]) -> Iterable[TT]:
    return chain(*(chain.from_iterable(aa.values()) for aa in args))


def map_layers(
        elements: Mapping[layer_t, Sequence[TT]],
        map_layer: Callable[[layer_t], layer_t],
        ) -> defaultdict[layer_t, list[TT]]:
    new_elements: defaultdict[layer_t, list[TT]] = defaultdict(list)
    for old_layer, seq in elements.items():
        new_layer = map_layer(old_layer)
        new_elements[new_layer].extend(seq)
    return new_elements


def map_targets(
        refs: Mapping[str | None, Sequence[Ref]],
        map_target: Callable[[str | None], str | None],
        ) -> defaultdict[str | None, list[Ref]]:
    new_refs: defaultdict[str | None, list[Ref]] = defaultdict(list)
    for old_target, seq in refs.items():
        new_target = map_target(old_target)
        new_refs[new_target].extend(seq)
    return new_refs
