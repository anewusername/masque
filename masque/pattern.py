"""
  Object representing a one multi-layer lithographic layout.
  A single level of hierarchical references is included.
"""
from typing import Callable, Sequence, cast, Mapping, Self, Any, Iterable, TypeVar, MutableMapping
import copy
import logging
from itertools import chain
from collections import defaultdict

import numpy
from numpy import inf, pi, nan
from numpy.typing import NDArray, ArrayLike
# .visualize imports matplotlib and matplotlib.collections

from .ref import Ref
from .abstract import Abstract
from .shapes import Shape, Polygon, Path, DEFAULT_POLY_NUM_VERTICES
from .label import Label
from .utils import rotation_matrix_2d, annotations_t, layer_t
from .error import PatternError, PortError
from .traits import AnnotatableImpl, Scalable, Mirrorable, Rotatable, Positionable, Repeatable, Bounded
from .ports import Port, PortList


logger = logging.getLogger(__name__)


class Pattern(PortList, AnnotatableImpl, Mirrorable):
    """
      2D layout consisting of some set of shapes, labels, and references to other
    Pattern objects (via Ref). Shapes are assumed to inherit from `masque.shapes.Shape`
    or provide equivalent functions.

    `Pattern` also stores a dict of `Port`s, which can be used to "snap" together points.
    See `Pattern.plug()` and `Pattern.place()`, as well as the helper classes
    `builder.Builder`, `builder.Pather`, `builder.RenderPather`, and `ports.PortsList`.

    For convenience, ports can be read out using square brackets:
    - `pattern['A'] == Port((0, 0), 0)`
    - `pattern[['A', 'B']] == {'A': Port((0, 0), 0), 'B': Port((0, 0), pi)}`


    Examples: Making a Pattern
    ==========================
    - `pat = Pattern()` just creates an empty pattern, with no geometry or ports

    - To immediately set some of the pattern's contents,
      ```
        pat = Pattern(
            shapes={'layer1': [shape0, ...], 'layer2': [shape,...], ...},
            labels={'layer1': [...], ...},
            refs={'name1': [ref0, ...], 'name2': [ref, ...], ...},
            ports={'in': Port(...), 'out': Port(...)},
            )
      ```

    - `Pattern.interface(other_pat, port_map=['A', 'B'])` makes a new
        (empty) pattern, copies over ports 'A' and 'B' from `other_pat`, and
        creates additional ports 'in_A' and 'in_B' facing in the opposite
        directions. This can be used to build a device which can plug into
        `other_pat` (using the 'in_*' ports) but which does not itself include
        `other_pat` as a subcomponent.


    Examples: Adding to a pattern
    =============================
    - `pat.plug(subdevice, {'A': 'C', 'B': 'B'}, map_out={'D': 'myport'})`
        instantiates `subdevice` into `pat`, plugging ports 'A' and 'B'
        of `pat` into ports 'C' and 'B' of `subdevice`. The connected ports
        are removed and any unconnected ports from `subdevice` are added to
        `pat`. Port 'D' of `subdevice` (unconnected) is renamed to 'myport'.

    - `pat.plug(wire, {'myport': 'A'})` places port 'A' of `wire` at 'myport'
        of `pat`. If `wire` has only two ports (e.g. 'A' and 'B'), since no `map_out`
        argument is provided and the `inherit_name` argument is not explicitly
        set to `False`, the unconnected port of `wire` is automatically renamed to
        'myport'. This allows easy extension of existing ports without changing
        their names or having to provide `map_out` each time `plug` is called.

    - `pat.place(pad, offset=(10, 10), rotation=pi / 2, port_map={'A': 'gnd'})`
        instantiates `pad` at the specified (x, y) offset and with the specified
        rotation, adding its ports to those of `pat`. Port 'A' of `pad` is
        renamed to 'gnd' so that further routing can use this signal or net name
        rather than the port name on the original `pad` device.
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
            cache: MutableMapping[str, NDArray[numpy.float64] | None] | None = None,
            ) -> NDArray[numpy.float64] | None:
        """
        Return a `numpy.ndarray` containing `[[x_min, y_min], [x_max, y_max]]`, corresponding to the
         extent of the Pattern's contents in each dimension.
        Returns `None` if the Pattern is empty.

        Args:
            library: If `recurse=True`, any referenced patterns are loaded from this library.
            recurse: If `False`, do not evaluate the bounds of any refs (i.e. assume they are empty).
                If `True`, evaluate the bounds of all refs and their conained geometry recursively.
                Default `True`.
            cache: Mapping of `{name: bounds}` for patterns for which the bounds have already been calculated.
                Modified during the run (any referenced pattern's bounds are added).

        Returns:
            `[[x_min, y_min], [x_max, y_max]]` or `None`
        """
        if self.is_empty():
            return None

        n_elems = sum(1 for _ in chain_elements(self.shapes, self.labels))
        ebounds = numpy.full((n_elems, 2, 2), nan)
        for ee, entry in enumerate(chain_elements(self.shapes, self.labels)):
            maybe_ebounds = cast(Bounded, entry).get_bounds()
            if maybe_ebounds is not None:
                ebounds[ee] = maybe_ebounds
        mask = ~numpy.isnan(ebounds[:, 0, 0])

        if mask.any():
            cbounds = numpy.vstack((
                numpy.min(ebounds[mask, 0, :], axis=0),
                numpy.max(ebounds[mask, 1, :], axis=0),
                ))
        else:
            cbounds = numpy.array((
                (+inf, +inf),
                (-inf, -inf),
                ))

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
                elif any(numpy.isclose(ref.rotation % (pi / 2), 0) for ref in refs):
                    unrot_bounds = library[target].get_bounds(library=library, recurse=recurse, cache=cache)
                    cache[target] = unrot_bounds

                for ref in refs:
                    if numpy.isclose(ref.rotation % (pi / 2), 0):
                        if unrot_bounds is None:
                            bounds = None
                        else:
                            ubounds = unrot_bounds.copy()
                            if ref.mirrored:
                                ubounds[:, 1] *= -1

                            corners = (rotation_matrix_2d(ref.rotation) @ ubounds.T).T
                            bounds = numpy.vstack((numpy.min(corners, axis=0),
                                                   numpy.max(corners, axis=0))) * ref.scale + [ref.offset]

                    else:
                        # Non-manhattan rotation, have to figure out bounds by rotating the pattern
                        bounds = ref.get_bounds(library[target], library=library)

                    if bounds is None:
                        continue

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
            library: If `recurse=True`, any referenced patterns are loaded from this library.
            recurse: If `False`, do not evaluate the bounds of any refs (i.e. assume they are empty).
                If `True`, evaluate the bounds of all refs and their conained geometry recursively.
                Default `True`.
            cache: Mapping of `{name: bounds}` for patterns for which the bounds have already been calculated.
                Modified during the run (any referenced pattern's bounds are added).

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

    def mirror_element_centers(self, across_axis: int = 0) -> Self:
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

    def mirror_elements(self, across_axis: int = 0) -> Self:
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

    def mirror(self, across_axis: int = 0) -> Self:
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
        Convenience method for `copy.deepcopy(pattern)` (same as `Pattern.deepcopy()`).
        See also: `Pattern.deepcopy()`

        Returns:
            A deep copy of the current Pattern.
        """
        return copy.deepcopy(self)

    def deepcopy(self) -> Self:
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
        return not (self.has_refs() or self.has_shapes() or self.has_labels())

    def has_refs(self) -> bool:
        """
        Returns:
            True if the pattern contains any refs.
        """
        return any(True for _ in chain.from_iterable(self.refs.values()))

    def has_shapes(self) -> bool:
        """
        Returns:
            True if the pattern contains any shapes.
        """
        return any(True for _ in chain.from_iterable(self.shapes.values()))

    def has_labels(self) -> bool:
        """
        Returns:
            True if the pattern contains any labels.
        """
        return any(True for _ in chain.from_iterable(self.labels.values()))

    def has_ports(self) -> bool:
        """
        Returns:
            True if the pattern contains any ports.
        """
        return bool(self.ports)

    def ref(self, target: str | Abstract | None, *args: Any, **kwargs: Any) -> Self:
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
        if isinstance(target, Abstract):
            target = target.name
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

    def prune_layers(self) -> Self:
        """
        Remove empty layers (empty lists) in `self.shapes` and `self.labels`.

        Returns:
            self
        """
        for layer in list(self.shapes):
            if not self.shapes[layer]:
                del self.shapes[layer]
        for layer in list(self.labels):
            if not self.labels[layer]:
                del self.labels[layer]
        return self

    def prune_refs(self) -> Self:
        """
        Remove empty ref lists in `self.refs`.

        Returns:
            self
        """
        for target in list(self.refs):
            if not self.refs[target]:
                del self.refs[target]
        return self

    def flatten(
            self,
            library: Mapping[str, 'Pattern'],
            flatten_ports: bool = False,
            ) -> 'Pattern':
        """
        Removes all refs (recursively) and adds equivalent shapes.
        Alters the current pattern in-place.
        For a version which creates copies, see `Library.flatten`.

        Args:
            library: Source for referenced patterns.
            flatten_ports: If `True`, keep ports from any referenced
                patterns; otherwise discard them.

        Returns:
            self
        """
        flattened: dict[str | None, 'Pattern | None'] = {}

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
        try:
            from matplotlib import pyplot       # type: ignore
            import matplotlib.collections       # type: ignore
        except ImportError as err:
            logger.error('Pattern.visualize() depends on matplotlib!')
            logger.error('Make sure to install masque with the [visualize] option to pull in the needed dependencies.')
            raise err

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

#    @overload
#    def place(
#            self,
#            other: 'Pattern',
#            *,
#            offset: ArrayLike,
#            rotation: float,
#            pivot: ArrayLike,
#            mirrored: bool,
#            port_map: dict[str, str | None] | None,
#            skip_port_check: bool,
#            append: bool,
#            ) -> Self:
#        pass
#
#    @overload
#    def place(
#            self,
#            other: Abstract,
#            *,
#            offset: ArrayLike,
#            rotation: float,
#            pivot: ArrayLike,
#            mirrored: bool,
#            port_map: dict[str, str | None] | None,
#            skip_port_check: bool,
#            append: Literal[False],
#            ) -> Self:
#        pass

    def place(
            self,
            other: 'Abstract | Pattern',
            *,
            offset: ArrayLike = (0, 0),
            rotation: float = 0,
            pivot: ArrayLike = (0, 0),
            mirrored: bool = False,
            port_map: dict[str, str | None] | None = None,
            skip_port_check: bool = False,
            append: bool = False,
            ) -> Self:
        """
        Instantiate or append the pattern `other` into the current pattern, adding its
          ports to those of the current pattern (but not connecting/removing any ports).

        Mirroring is applied before rotation; translation (`offset`) is applied last.

        Examples:
        =========
        - `my_pat.place(pad_pat, offset=(10, 10), rotation=pi / 2, port_map={'A': 'gnd'})`
            instantiates `pad` at the specified (x, y) offset and with the specified
            rotation, adding its ports to those of `my_pat`. Port 'A' of `pad` is
            renamed to 'gnd' so that further routing can use this signal or net name
            rather than the port name on the original `pad_pat` pattern.

        Args:
            other: An `Abstract` or `Pattern` describing the device to be instatiated.
            offset: Offset at which to place the instance. Default (0, 0).
            rotation: Rotation applied to the instance before placement. Default 0.
            pivot: Rotation is applied around this pivot point (default (0, 0)).
                Rotation is applied prior to translation (`offset`).
            mirrored: Whether theinstance should be mirrored across the x axis.
                Mirroring is applied before translation and rotation.
            port_map: dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for ports in the instantiated pattern. New names can be
                `None`, which will delete those ports.
            skip_port_check: Can be used to skip the internal call to `check_ports`,
                in case it has already been performed elsewhere.
            append: If `True`, `other` is appended instead of being referenced.
                Note that this does not flatten  `other`, so its refs will still
                be refs (now inside `self`).

        Returns:
            self

        Raises:
            `PortError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `other.ports`.
            `PortError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
        """
        if port_map is None:
            port_map = {}

        if not skip_port_check:
            self.check_ports(other.ports.keys(), map_in=None, map_out=port_map)

        ports = {}
        for name, port in other.ports.items():
            new_name = port_map.get(name, name)
            if new_name is None:
                continue
            ports[new_name] = port

        for name, port in ports.items():
            p = port.deepcopy()
            if mirrored:
                p.mirror()
            p.rotate_around(pivot, rotation)
            p.translate(offset)
            self.ports[name] = p

        if append:
            if isinstance(other, Abstract):
                raise PatternError('Must provide a full `Pattern` (not an `Abstract`) when appending!')
            other_copy = other.deepcopy()
            other_copy.ports.clear()
            if mirrored:
                other_copy.mirror()
            other_copy.rotate_around(pivot, rotation)
            other_copy.translate_elements(offset)
            self.append(other_copy)
        else:
            assert not isinstance(other, Pattern)
            ref = Ref(mirrored=mirrored)
            ref.rotate_around(pivot, rotation)
            ref.translate(offset)
            self.refs[other.name].append(ref)
        return self

#    @overload
#    def plug(
#            self,
#            other: Abstract,
#            map_in: dict[str, str],
#            map_out: dict[str, str | None] | None,
#            *,
#            mirrored: bool,
#            inherit_name: bool,
#            set_rotation: bool | None,
#            append: Literal[False],
#            ) -> Self:
#        pass
#
#    @overload
#    def plug(
#            self,
#            other: 'Pattern',
#            map_in: dict[str, str],
#            map_out: dict[str, str | None] | None,
#            *,
#            mirrored: bool,
#            inherit_name: bool,
#            set_rotation: bool | None,
#            append: bool,
#            ) -> Self:
#        pass

    def plug(
            self,
            other: 'Abstract | Pattern',
            map_in: dict[str, str],
            map_out: dict[str, str | None] | None = None,
            *,
            mirrored: bool = False,
            inherit_name: bool = True,
            set_rotation: bool | None = None,
            append: bool = False,
            ) -> Self:
        """
        Instantiate or append a pattern into the current pattern, connecting
          the ports specified by `map_in` and renaming the unconnected
          ports specified by `map_out`.

        Examples:
        =========
        - `my_pat.plug(subdevice, {'A': 'C', 'B': 'B'}, map_out={'D': 'myport'})`
            instantiates `subdevice` into `my_pat`, plugging ports 'A' and 'B'
            of `my_pat` into ports 'C' and 'B' of `subdevice`. The connected ports
            are removed and any unconnected ports from `subdevice` are added to
            `my_pat`. Port 'D' of `subdevice` (unconnected) is renamed to 'myport'.

        - `my_pat.plug(wire, {'myport': 'A'})` places port 'A' of `wire` at 'myport'
            of `my_pat`.
            If `wire` has only two ports (e.g. 'A' and 'B'), no `map_out` argument is
            provided, and the `inherit_name` argument is not explicitly set to `False`,
            the unconnected port of `wire` is automatically renamed to 'myport'. This
            allows easy extension of existing ports without changing their names or
            having to provide `map_out` each time `plug` is called.

        Args:
            other: A `Pattern` or `Abstract` describing the subdevice to be instatiated.
            map_in: dict of `{'self_port': 'other_port'}` mappings, specifying
                port connections between the current pattern and the subdevice.
            map_out: dict of `{'old_name': 'new_name'}` mappings, specifying
                new names for ports in `other`.
            mirrored: Enables mirroring `other` across the x axis prior to connecting
                any ports.
            inherit_name: If `True`, and `map_in` specifies only a single port,
                and `map_out` is `None`, and `other` has only two ports total,
                then automatically renames the output port of `other` to the
                name of the port from `self` that appears in `map_in`. This
                makes it easy to extend a pattern with simple 2-port devices
                (e.g. wires) without providing `map_out` each time `plug` is
                called. See "Examples" above for more info. Default `True`.
            set_rotation: If the necessary rotation cannot be determined from
                the ports being connected (i.e. all pairs have at least one
                port with `rotation=None`), `set_rotation` must be provided
                to indicate how much `other` should be rotated. Otherwise,
                `set_rotation` must remain `None`.
            append: If `True`, `other` is appended instead of being referenced.
                Note that this does not flatten  `other`, so its refs will still
                be refs (now inside `self`).

        Returns:
            self

        Raises:
            `PortError` if any ports specified in `map_in` or `map_out` do not
                exist in `self.ports` or `other_names`.
            `PortError` if there are any duplicate names after `map_in` and `map_out`
                are applied.
            `PortError` if the specified port mapping is not achieveable (the ports
                do not line up)
        """
        # If asked to inherit a name, check that all conditions are met
        if (inherit_name
                and not map_out
                and len(map_in) == 1
                and len(other.ports) == 2):
            out_port_name = next(iter(set(other.ports.keys()) - set(map_in.values())))
            map_out = {out_port_name: next(iter(map_in.keys()))}

        if map_out is None:
            map_out = {}
        map_out = copy.deepcopy(map_out)

        self.check_ports(other.ports.keys(), map_in, map_out)
        translation, rotation, pivot = self.find_transform(
            other,
            map_in,
            mirrored=mirrored,
            set_rotation=set_rotation,
            )

        # get rid of plugged ports
        for ki, vi in map_in.items():
            del self.ports[ki]
            map_out[vi] = None

        if isinstance(other, Pattern):
            assert append

        self.place(
            other,
            offset=translation,
            rotation=rotation,
            pivot=pivot,
            mirrored=mirrored,
            port_map=map_out,
            skip_port_check=True,
            append=append,
            )
        return self

    @classmethod
    def interface(
            cls,
            source: PortList | Mapping[str, Port],
            *,
            in_prefix: str = 'in_',
            out_prefix: str = '',
            port_map: dict[str, str] | Sequence[str] | None = None,
            ) -> 'Pattern':
        """
        Generate an empty pattern with ports based on all or some of the ports
          in the `source`. Do not include the source device istelf; instead use
          it to define ports (the "interface") for the new device.

        The ports specified by `port_map` (default: all ports) are copied to
          new device, and additional (input) ports are created facing in the
          opposite directions. The specified `in_prefix` and `out_prefix` are
          prepended to the port names to differentiate them.

        By default, the flipped ports are given an 'in_' prefix and unflipped
          ports keep their original names, enabling intuitive construction of
          a device that will "plug into" the current device; the 'in_*' ports
          are used for plugging the devices together while the original port
          names are used for building the new device.

        Another use-case could be to build the new device using the 'in_'
          ports, creating a new device which could be used in place of the
          current device.

        Args:
            source: A collection of ports (e.g. Pattern, Builder, or dict)
                from which to create the interface.
            in_prefix: Prepended to port names for newly-created ports with
                reversed directions compared to the current device.
            out_prefix: Prepended to port names for ports which are directly
                copied from the current device.
            port_map: Specification for ports to copy into the new device:
                - If `None`, all ports are copied.
                - If a sequence, only the listed ports are copied
                - If a mapping, the listed ports (keys) are copied and
                    renamed (to the values).

        Returns:
            The new empty pattern, with 2x as many ports as listed in port_map.

        Raises:
            `PortError` if `port_map` contains port names not present in the
                current device.
            `PortError` if applying the prefixes results in duplicate port
                names.
        """
        if isinstance(source, PortList):
            orig_ports = source.ports
        elif isinstance(source, dict):
            orig_ports = source
        else:
            raise PatternError(f'Unable to get ports from {type(source)}: {source}')

        if port_map:
            if isinstance(port_map, dict):
                missing_inkeys = set(port_map.keys()) - set(orig_ports.keys())
                mapped_ports = {port_map[k]: v for k, v in orig_ports.items() if k in port_map}
            else:
                port_set = set(port_map)
                missing_inkeys = port_set - set(orig_ports.keys())
                mapped_ports = {k: v for k, v in orig_ports.items() if k in port_set}

            if missing_inkeys:
                raise PortError(f'`port_map` keys not present in source: {missing_inkeys}')
        else:
            mapped_ports = orig_ports

        ports_in = {f'{in_prefix}{name}': port.deepcopy().rotate(pi)
                    for name, port in mapped_ports.items()}
        ports_out = {f'{out_prefix}{name}': port.deepcopy()
                     for name, port in mapped_ports.items()}

        duplicates = set(ports_out.keys()) & set(ports_in.keys())
        if duplicates:
            raise PortError(f'Duplicate keys after prefixing, try a different prefix: {duplicates}')

        new = Pattern(ports={**ports_in, **ports_out})
        return new


TT = TypeVar('TT')


def chain_elements(*args: Mapping[Any, Iterable[TT]]) -> Iterable[TT]:
    """
    Iterate over each element in one or more {layer: elements} mappings.

    Useful when you want to do some operation on all shapes and/or labels,
    disregarding which layer they are on.

    Args:
        *args: One or more {layer: [element0, ...]} mappings.
            Can also be applied to e.g. {target: [ref0, ...]} mappings.

    Returns:
        An iterable containing all elements, regardless of layer.
    """
    return chain(*(chain.from_iterable(aa.values()) for aa in args))


def map_layers(
        elements: Mapping[layer_t, Sequence[TT]],
        map_layer: Callable[[layer_t], layer_t],
        ) -> defaultdict[layer_t, list[TT]]:
    """
    Move all the elements from one layer onto a different layer.
    Can also handle multiple such mappings simultaneously.

    Args:
        elements: Mapping of {old_layer: geometry_or_labels}.
        map_layer: Callable which may be called with each layer present in `elements`,
            and should return the new layer to which it will be mapped.
            A simple example which maps `old_layer` to `new_layer` and leaves all others
            as-is would look like `lambda layer: {old_layer: new_layer}.get(layer, layer)`

    Returns:
        Mapping of {new_layer: geometry_or_labels}
    """
    new_elements: defaultdict[layer_t, list[TT]] = defaultdict(list)
    for old_layer, seq in elements.items():
        new_layer = map_layer(old_layer)
        new_elements[new_layer].extend(seq)
    return new_elements


def map_targets(
        refs: Mapping[str | None, Sequence[Ref]],
        map_target: Callable[[str | None], str | None],
        ) -> defaultdict[str | None, list[Ref]]:
    """
    Change the target of all references to a given cell.
    Can also handle multiple such mappings simultaneously.

    Args:
        refs: Mapping of {old_target: ref_objects}.
        map_target: Callable which may be called with each target present in `refs`,
            and should return the new target to which it will be mapped.
            A simple example which maps `old_target` to `new_target` and leaves all others
            as-is would look like `lambda target: {old_target: new_target}.get(target, target)`

    Returns:
        Mapping of {new_target: ref_objects}
    """
    new_refs: defaultdict[str | None, list[Ref]] = defaultdict(list)
    for old_target, seq in refs.items():
        new_target = map_target(old_target)
        new_refs[new_target].extend(seq)
    return new_refs
