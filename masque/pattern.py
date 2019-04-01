"""
 Base object for containing a lithography mask.
"""

from typing import List, Callable, Tuple, Dict, Union
import copy
import itertools
import pickle
from collections import defaultdict

import numpy
# .visualize imports matplotlib and matplotlib.collections

from .subpattern import SubPattern
from .repetition import GridRepetition
from .shapes import Shape, Polygon
from .label import Label
from .utils import rotation_matrix_2d, vector2
from .error import PatternError

__author__ = 'Jan Petykiewicz'


class Pattern:
    """
    2D layout consisting of some set of shapes and references to other Pattern objects
     (via SubPattern). Shapes are assumed to inherit from .shapes.Shape or provide equivalent
     functions.

    :var shapes: List of all shapes in this Pattern. Elements in this list are assumed to inherit
            from Shape or provide equivalent functions.
    :var subpatterns: List of all SubPattern objects in this Pattern. Multiple SubPattern objects
            may reference the same Pattern object.
    :var name: An identifier for this object. Not necessarily unique.
    """
    shapes = None               # type: List[Shape]
    labels = None               # type: List[Labels]
    subpatterns = None          # type: List[SubPattern or GridRepetition]
    name = None                 # type: str

    def __init__(self,
                 shapes: List[Shape]=(),
                 labels: List[Label]=(),
                 subpatterns: List[SubPattern]=(),
                 name: str='',
                 ):
        """
        Basic init; arguments get assigned to member variables.
         Non-list inputs for shapes and subpatterns get converted to lists.

        :param shapes: Initial shapes in the Pattern
        :param labels: Initial labels in the Pattern
        :param subpatterns: Initial subpatterns in the Pattern
        :param name: An identifier for the Pattern
        """
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

        self.name = name

    def append(self, other_pattern: 'Pattern') -> 'Pattern':
        """
        Appends all shapes, labels and subpatterns from other_pattern to self's shapes,
          labels, and supbatterns.

        :param other_pattern: The Pattern to append
        :return: self
        """
        self.subpatterns += other_pattern.subpatterns
        self.shapes += other_pattern.shapes
        self.labels += other_pattern.labels
        return self

    def subset(self,
               shapes_func: Callable[[Shape], bool]=None,
               labels_func: Callable[[Label], bool]=None,
               subpatterns_func: Callable[[SubPattern], bool]=None,
               recursive: bool=False,
               ) -> 'Pattern':
        """
        Returns a Pattern containing only the entities (e.g. shapes) for which the
          given entity_func returns True.
        Self is _not_ altered, but shapes, labels, and subpatterns are _not_ copied.

        :param shapes_func: Given a shape, returns a boolean denoting whether the shape is a member
             of the subset. Default always returns False.
        :param labels_func: Given a label, returns a boolean denoting whether the label is a member
             of the subset. Default always returns False.
        :param subpatterns_func: Given a subpattern, returns a boolean denoting if it is a member
             of the subset. Default always returns False.
        :param recursive: If True, also calls .subset() recursively on patterns referenced by this
             pattern.
        :return: A Pattern containing all the shapes and subpatterns for which the parameter
             functions return True
        """
        def do_subset(src):
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
        return pat

    def apply(self,
              func: Callable[['Pattern'], 'Pattern'],
              memo: Dict[int, 'Pattern']=None,
              ) -> 'Pattern':
        """
        Recursively apply func() to this pattern and any pattern it references.
        func() is expected to take and return a Pattern.
        func() is first applied to the pattern as a whole, then any referenced patterns.
        It is only applied to any given pattern once, regardless of how many times it is
            referenced.

        :param func: Function which accepts a Pattern, and returns a pattern.
        :param memo: Dictionary used to avoid re-running on multiply-referenced patterns.
            Stores {id(pattern): func(pattern)} for patterns which have already been processed.
            Default None (no already-processed patterns).
        :return: The result of applying func() to this pattern and all subpatterns.
        :raises: PatternError if called on a pattern containing a circular reference.
        """
        if memo is None:
            memo = {}

        pat_id = id(self)
        if pat_id not in memo:
            memo[pat_id] = None
            pat = func(self)
            for subpat in pat.subpatterns:
                subpat.pattern = subpat.pattern.apply(func, memo)
            memo[pat_id] = pat
        elif memo[pat_id] is None:
            raise PatternError('.apply() called on pattern with circular reference')
        else:
            pat = memo[pat_id]
        return pat

    def polygonize(self,
                   poly_num_points: int=None,
                   poly_max_arclen: float=None
                   ) -> 'Pattern':
        """
        Calls .to_polygons(...) on all the shapes in this Pattern and any referenced patterns,
         replacing them with the returned polygons.
        Arguments are passed directly to shape.to_polygons(...).

        :param poly_num_points: Number of points to use for each polygon. Can be overridden by
             poly_max_arclen if that results in more points. Optional, defaults to shapes'
             internal defaults.
        :param poly_max_arclen: Maximum arclength which can be approximated by a single line
             segment. Optional, defaults to shapes' internal defaults.
        :return: self
        """
        old_shapes = self.shapes
        self.shapes = list(itertools.chain.from_iterable(
                        (shape.to_polygons(poly_num_points, poly_max_arclen)
                         for shape in old_shapes)))
        for subpat in self.subpatterns:
            subpat.pattern.polygonize(poly_num_points, poly_max_arclen)
        return self

    def manhattanize(self,
                     grid_x: numpy.ndarray,
                     grid_y: numpy.ndarray
                     ) -> 'Pattern':
        """
        Calls .polygonize() and .flatten on the pattern, then calls .manhattanize() on all the
         resulting shapes, replacing them with the returned Manhattan polygons.

        :param grid_x: List of allowed x-coordinates for the Manhattanized polygon edges.
        :param grid_y: List of allowed y-coordinates for the Manhattanized polygon edges.
        :return: self
        """

        self.polygonize().flatten()
        old_shapes = self.shapes
        self.shapes = list(itertools.chain.from_iterable(
                        (shape.manhattanize(grid_x, grid_y) for shape in old_shapes)))
        return self

    def subpatternize(self,
                      recursive: bool=True,
                      norm_value: int=1e6,
                      exclude_types: Tuple[Shape]=(Polygon,)
                      ) -> 'Pattern':
        """
        Iterates through this Pattern and all referenced Patterns. Within each Pattern, it iterates
         over all shapes, calling .normalized_form(norm_value) on them to retrieve a scale-,
         offset-, dose-, and rotation-independent form. Each shape whose normalized form appears
         more than once is removed and re-added using subpattern objects referencing a newly-created
         Pattern containing only the normalized form of the shape.

        Note that the default norm_value was chosen to give a reasonable precision when converting
         to GDSII, which uses integer values for pixel coordinates.

        :param recursive: Whether to call recursively on self's subpatterns. Default True.
        :param norm_value: Passed to shape.normalized_form(norm_value). Default 1e6 (see function
                note about GDSII)
        :param exclude_types: Shape types passed in this argument are always left untouched, for
                speed or convenience. Default: (Shapes.Polygon,)
        :return: self
        """

        if exclude_types is None:
            exclude_types = ()

        if recursive:
            for subpat in self.subpatterns:
                subpat.pattern.subpatternize(recursive=True,
                                             norm_value=norm_value,
                                             exclude_types=exclude_types)

        # Create a dict which uses the label tuple from .normalized_form() as a key, and which
        #  stores (function_to_create_normalized_shape, [(index_in_shapes, values), ...]), where
        #  values are the (offset, scale, rotation, dose) values as calculated by .normalized_form()
        shape_table = defaultdict(lambda: [None, list()])
        for i, shape in enumerate(self.shapes):
            if not any((isinstance(shape, t) for t in exclude_types)):
                label, values, func = shape.normalized_form(norm_value)
                shape_table[label][0] = func
                shape_table[label][1].append((i, values))

        # Iterate over the normalized shapes in the table. If any normalized shape occurs more than
        #  once, create a Pattern holding a normalized shape object, and add self.subpatterns
        #  entries for each occurrence in self. Also, note down that we should delete the
        #  self.shapes entries for which we made SubPatterns.
        shapes_to_remove = []
        for label in shape_table:
            if len(shape_table[label][1]) > 1:
                shape = shape_table[label][0]()
                pat = Pattern(shapes=[shape])

                for i, values in shape_table[label][1]:
                    (offset, scale, rotation, dose) = values
                    subpat = SubPattern(pattern=pat, offset=offset, scale=scale,
                                        rotation=rotation, dose=dose)
                    self.subpatterns.append(subpat)
                    shapes_to_remove.append(i)

        # Remove any shapes for which we have created subpatterns.
        for i in sorted(shapes_to_remove, reverse=True):
            del self.shapes[i]

        return self

    def as_polygons(self) -> List[numpy.ndarray]:
        """
        Represents the pattern as a list of polygons.

        Deep-copies the pattern, then calls .polygonize() and .flatten() on the copy in order to
         generate the list of polygons.

        :return: A list of (Ni, 2) numpy.ndarrays specifying vertices of the polygons. Each ndarray
            is of the form [[x0, y0], [x1, y1],...].
        """
        pat = copy.deepcopy(self).polygonize().flatten()
        return [shape.vertices + shape.offset for shape in pat.shapes]

    def referenced_patterns_by_id(self) -> Dict[int, 'Pattern']:
        """
        Create a dictionary of {id(pat): pat} for all Pattern objects referenced by this
         Pattern (operates recursively on all referenced Patterns as well)

        :return: Dictionary of {id(pat): pat} for all referenced Pattern objects
        """
        ids = {}
        for subpat in self.subpatterns:
            if id(subpat.pattern) not in ids:
                ids[id(subpat.pattern)] = subpat.pattern
                ids.update(subpat.pattern.referenced_patterns_by_id())
        return ids

    def get_bounds(self) -> Union[numpy.ndarray, None]:
        """
        Return a numpy.ndarray containing [[x_min, y_min], [x_max, y_max]], corresponding to the
         extent of the Pattern's contents in each dimension.
        Returns None if the Pattern is empty.

        :return: [[x_min, y_min], [x_max, y_max]] or None
        """
        entries = self.shapes + self.subpatterns + self.labels
        if not entries:
            return None

        init_bounds = entries[0].get_bounds()
        min_bounds = init_bounds[0, :]
        max_bounds = init_bounds[1, :]
        for entry in entries[1:]:
            bounds = entry.get_bounds()
            min_bounds = numpy.minimum(min_bounds, bounds[0, :])
            max_bounds = numpy.maximum(max_bounds, bounds[1, :])
        return numpy.vstack((min_bounds, max_bounds))

    def flatten(self) -> 'Pattern':
        """
        Removes all subpatterns and adds equivalent shapes.

        :return: self
        """
        subpatterns = copy.deepcopy(self.subpatterns)
        self.subpatterns = []
        for subpat in subpatterns:
            subpat.pattern.flatten()
            p = subpat.as_pattern()
            self.shapes += p.shapes
            self.labels += p.labels
        return self

    def translate_elements(self, offset: vector2) -> 'Pattern':
        """
        Translates all shapes, label, and subpatterns by the given offset.

        :param offset: Offset to translate by
        :return: self
        """
        for entry in self.shapes + self.subpatterns + self.labels:
            entry.translate(offset)
        return self

    def scale_elements(self, scale: float) -> 'Pattern':
        """"
        Scales all shapes and subpatterns by the given value.

        :param scale: value to scale by
        :return: self
        """
        for entry in self.shapes + self.subpatterns:
            entry.scale(scale)
        return self

    def scale_by(self, c: float) -> 'Pattern':
        """
        Scale this Pattern by the given value
         (all shapes and subpatterns and their offsets are scaled)

        :param c: value to scale by
        :return: self
        """
        for entry in self.shapes + self.subpatterns:
            entry.offset *= c
            entry.scale_by(c)
        return self

    def rotate_around(self, pivot: vector2, rotation: float) -> 'Pattern':
        """
        Rotate the Pattern around the a location.

        :param pivot: Location to rotate around
        :param rotation: Angle to rotate by (counter-clockwise, radians)
        :return: self
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

        :param rotation: Angle to rotate by (counter-clockwise, radians)
        :return: self
        """
        for entry in self.shapes + self.subpatterns + self.labels:
            entry.offset = numpy.dot(rotation_matrix_2d(rotation), entry.offset)
        return self

    def rotate_elements(self, rotation: float) -> 'Pattern':
        """
        Rotate each shape and subpattern around its center (offset)

        :param rotation: Angle to rotate by (counter-clockwise, radians)
        :return: self
        """
        for entry in self.shapes + self.subpatterns:
            entry.rotate(rotation)
        return self

    def mirror_element_centers(self, axis: int) -> 'Pattern':
        """
        Mirror the offsets of all shapes, labels, and subpatterns across an axis

        :param axis: Axis to mirror across
        :return: self
        """
        for entry in self.shapes + self.subpatterns + self.labels:
            entry.offset[axis - 1] *= -1
        return self

    def mirror_elements(self, axis: int) -> 'Pattern':
        """
        Mirror each shape and subpattern across an axis, relative to its
          center (offset)

        :param axis: Axis to mirror across
        :return: self
        """
        for entry in self.shapes + self.subpatterns:
            entry.mirror(axis)
        return self

    def mirror(self, axis: int) -> 'Pattern':
        """
        Mirror the Pattern across an axis

        :param axis: Axis to mirror across
        :return: self
        """
        self.mirror_elements(axis)
        self.mirror_element_centers(axis)
        return self

    def scale_element_doses(self, factor: float) -> 'Pattern':
        """
        Multiply all shape and subpattern doses by a factor

        :param factor: Factor to multiply doses by
        :return: self
        """
        for entry in self.shapes + self.subpatterns:
            entry.dose *= factor
        return self

    def copy(self) -> 'Pattern':
        """
        Return a copy of the Pattern, deep-copying shapes and copying subpattern entries, but not
         deep-copying any referenced patterns.

	See also: Pattern.deepcopy()

        :return: A copy of the current Pattern.
        """
        cp = copy.copy(self)
        cp.shapes = copy.deepcopy(cp.shapes)
        cp.labels = copy.deepcopy(cp.labels)
        cp.subpatterns = [copy.copy(subpat) for subpat in cp.subpatterns]
        return cp

    def deepcopy(self) -> 'Pattern':
        """
        Convenience method for copy.deepcopy(pattern)

        :return: A deep copy of the current Pattern.
        """
        return copy.deepcopy(self)

    @staticmethod
    def load(filename: str) -> 'Pattern':
        """
        Load a Pattern from a file

        :param filename: Filename to load from
        :return: Loaded Pattern
        """
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)

        pattern = Pattern()
        pattern.__dict__.update(tmp_dict)
        return pattern

    def save(self, filename: str) -> 'Pattern':
        """
        Save the Pattern to a file

        :param filename: Filename to save to
        :return: self
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, protocol=2)
        return self

    def visualize(self,
                  offset: vector2=(0., 0.),
                  line_color: str='k',
                  fill_color: str='none',
                  overdraw: bool=False):
        """
        Draw a picture of the Pattern and wait for the user to inspect it

        Imports matplotlib.

        :param offset: Coordinates to offset by before drawing
        :param line_color: Outlines are drawn with this color (passed to matplotlib PolyCollection)
        :param fill_color: Interiors are drawn with this color (passed to matplotlib PolyCollection)
        :param overdraw: Whether to create a new figure or draw on a pre-existing one
        """
        # TODO: add text labels to visualize()
        from matplotlib import pyplot
        import matplotlib.collections

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
