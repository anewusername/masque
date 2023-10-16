# Masque README

Masque is a Python module for designing lithography masks.

The general idea is to implement something resembling the GDSII file-format, but
with some vectorized element types (eg. circles, not just polygons) and the ability
to output to multiple formats.

- [Source repository](https://mpxd.net/code/jan/masque)
- [PyPI](https://pypi.org/project/masque)


## Installation

Requirements:
* python >= 3.11
* numpy
* klamath (used for GDSII i/o)

Optional requirements:
* `ezdxf` (DXF i/o): ezdxf
* `oasis` (OASIS i/o): fatamorgana
* `svg` (SVG output): svgwrite
* `visualization` (shape plotting): matplotlib
* `text` (`Text` shape): matplotlib, freetype


Install with pip:
```bash
pip install 'masque[oasis,dxf,svg,visualization,text]'
```

## Overview

A layout consists of a hierarchy of `Pattern`s stored in a single `Library`.
Each `Pattern` can contain `Ref`s pointing at other patterns, `Shape`s, `Label`s, and `Port`s.


`masque` departs from several "classic" GDSII paradigms:
- A `Pattern` object does not store its own name. A name is only assigned when the pattern is placed
    into a `Library`, which is effectively a name->`Pattern` mapping.
- Layer info for `Shape`ss and `Label`s is not stored in the individual shape and label objects.
    Instead, the layer is determined by the key for the container dict (e.g. `pattern.shapes[layer]`).
    * This simplifies many common tasks: filtering `Shape`s by layer, remapping layers, and checking if
        a layer is empty.
    * Technically, this allows reusing the same shape or label object across multiple layers. This isn't
        part of the standard workflow since a mixture of single-use and multi-use shapes could be confusing.
    * This is similar to the approach used in [KLayout](https://www.klayout.de)
- `Ref` target names are also determined in the key of the container dict (e.g. `pattern.refs[target_name]`).
    * This similarly simplifies filtering `Ref`s by target name, updating to a new target, and checking
        if a given `Pattern` is referenced.
- `Pattern` names are set by their containing `Library` and are not stored in the `Pattern` objects.
    * This guarantees that there are no duplicate pattern names within any given `Library`.
    * Likewise, enumerating all the names (and all the `Pattern`s) in a `Library` is straightforward.
- Each `Ref`, `Shape`, or `Label` can be repeated multiple times by attaching a `repetition` object to it.
    * This is similar to how OASIS reptitions are handled, and provides extra flexibility over the GDSII
        approach of only allowing arrays through AREF (`Ref` + `repetition`).
- `Label`s do not have an orientation or presentation
    * This is in line with how they are used in practice, and how they are represented in OASIS.
- Non-polygonal `Shape`s are allowed. For example, elliptical arcs are a basic shape type.
    * This enables compatibility with OASIS (e.g. circles) and other formats.
    * `Shape`s provide a `.to_polygons()` method for GDSII compatibility.
- Most coordinate values are stored as 64-bit floats internally.
    * 1 earth radii in nanometers (6e15) is still represented without approximation (53 bit mantissa -> 2^53 > 9e15)
    * Operations that would otherwise clip/round on are still represented approximately.
    * Memory usage is usually dominated by other Python overhead.
- `Pattern` objects also contain `Port` information, which can be used to "snap" together
    multiple sub-components by matching up the requested port offsets and rotations.
    * Port rotations are defined as counter-clockwise angles from the +x axis.
    * Ports point into the interior of their associated device.
    * Port rotations may be `None` in the case of non-oriented ports.
    * Ports have a `ptype` string which is compared in order to catch mismatched connections at build time.
    * Ports can be exported into/imported from `Label`s stored directly in the layout,
        editable from standard tools (e.g. KLayout). A default format is provided.

In one important way, `masque` stays very orthodox:
References are accomplished by listing the target's name, not its `Pattern` object.

- The main downside of this is that any operations that traverse the hierarchy require
    both the `Pattern` and the `Library` which is contains its reference targets.
- This guarantees that names within a `Library` remain unique at all times.
    * Since this can be tedious in cases where you don't actually care about the name of a
        pattern, patterns whose names start with `SINGLE_USE_PREFIX` (default: an underscore)
        may be silently renamed in order to maintain uniqueness.
        See `masque.library.SINGLE_USE_PREFIX`, `masque.library._rename_patterns()`,
        and `ILibrary.add()` for more details.
- Having all patterns accessible through the `Library` avoids having to perform a
    tree traversal for every operation which needs to touch all `Pattern` objects
    (e.g. deleting a layer everywhere or scaling all patterns).
- Since `Pattern` doesn't know its own name, you can't create a reference by passing in
    a `Pattern` object -- you need to know its name.
- You *can* reference a `Pattern` before it is created, so long as you have already decided
    on its name.
- Functions like `Pattern.place()` and `Pattern.plug()` need to receive a pattern's name
    in order to create a reference, but they also need to access the pattern's ports.
    * One way to provide this data is through an `Abstract`, generated via
        `Library.abstract()` or through a `Library.abstract_view()`.
    * Another way is use `Builder.place()` or `Builder.plug()`, which automatically creates
        an `Abstract` from its internally-referenced `Library`.


## Glossary
- `Library`: A collection of named cells. OASIS or GDS "library" or file.
- "tree": Any Library which has only one topcell.
- `Pattern`: A collection of geometry, text labels, and reference to other patterns.
        OASIS or GDS "Cell", DXF "Block".
- `Ref`: A reference to another pattern. GDS "AREF/SREF", OASIS "Placement".
- `Shape`: Individual geometric entity. OASIS or GDS "Geometry element", DXF "LWPolyline" or "Polyline".
- `repetition`: Repetition operation. OASIS "repetition".
        GDS "AREF" is a `Ref` combined with a `Grid` repetition.
- `Label`: Text label. Not rendered into geometry. OASIS, GDS, DXF "Text".
- `annotation`: Additional metadata. OASIS or GDS "property".


## Syntax, shorthand, and design patterns
Most syntax and behavior should follow normal python conventions.
There are a few exceptions, either meant to catch common mistakes or to provide a shorthand for common operations:

### `Library` objects don't allow overwriting already-existing patterns
```python3
library['mycell'] = pattern0
library['mycell'] = pattern1   # Error! 'mycell' already exists and can't be overwritten
del library['mycell']          # We can explicitly delete it
library['mycell'] = pattern1   # And now it's ok to assign a new value
library.delete('mycell')       # This also deletes all refs pointing to 'mycell' by default
```

### Insert a newly-made hierarchical pattern (with children) into a layout
```python3
# Let's say we have a function which returns a new library containing one topcell (and possibly children)
tree = make_tree(...)

# To reference this cell in our layout, we have to add all its children to our `library` first:
top_name = tree.top()              # get the name of the topcell
name_mapping = library.add(tree)   # add all patterns from `tree`, renaming elgible conflicting patterns
new_name = name_mapping.get(top_name, top_name)    # get the new name for the cell (in case it was auto-renamed)
my_pattern.ref(new_name, ...)       # instantiate the cell

# This can be accomplished as follows
new_name = library << tree       # Add `tree` into `library` and return the top cell's new name
my_pattern.ref(new_name, ...)       # instantiate the cell

# In practice, you may do lots of
my_pattern.ref(lib << make_tree(...), ...)
```

We can also use this shorthand to quickly add and reference a single flat (as yet un-named) pattern:
```python3
anonymous_pattern = Pattern(...)
my_pattern.ref(lib << {'_tentative_name': anonymous_pattern}, ...)
```

### Place a hierarchical pattern into a layout, preserving its port info
```python3
# As above, we have a function that makes a new library containing one topcell (and possibly children)
tree = make_tree(...)

# We need to go get its port info to `place()` it into our existing layout,
new_name = library << tree          # Add the tree to the library and return its name (see `<<` above)
abstract = library.abstract(tree)   # An `Abstract` stores a pattern's name and its ports (but no geometry)
my_pattern.place(abstract, ...)

# With shorthand,
abstract = library <= tree
my_pattern.place(abstract, ...)

# or
my_pattern.place(library << make_tree(...), ...)


### Quickly add geometry, labels, or refs:
The long form for adding elements can be overly verbose:
```python3
my_pattern.shapes[layer].append(Polygon(vertices, ...))
my_pattern.labels[layer] += [Label('my text')]
my_pattern.refs[target_name].append(Ref(offset=..., ...))
```

There is shorthand for the most common elements:
```python3
my_pattern.polygon(layer=layer, vertices=vertices, ...)
my_pattern.rect(layer=layer, xctr=..., xmin=..., ymax=..., ly=...)  # rectangle; pick 4 of 6 constraints
my_pattern.rect(layer=layer, ymin=..., ymax=..., xctr=..., lx=...)
my_pattern.path(...)
my_pattern.label(layer, 'my_text')
my_pattern.ref(target_name, offset=..., ...)
```

### Accessing ports
```python3
# Square brackets pull from the underlying `.ports` dict:
assert pattern['input'] is pattern.ports['input']

# And you can use them to read multiple ports at once:
assert pattern[('input', 'output')] == {
    'input': pattern.ports['input'],
    'output': pattern.ports['output'],
    }

# But you shouldn't use them for anything except reading
pattern['input'] = Port(...)   # Error!
has_input = ('input' in pattern)   # Error!
```

### Building patterns
```python3
library = Library(...)
my_pattern_name, my_pattern = library.mkpat(some_name_generator())
...
def _make_my_subpattern() -> str:
    #   This function can draw from the outer scope (e.g. `library`) but will not pollute the outer scope
    # (e.g. the variable `subpattern` will not be accessible from outside the function; you must load it
    # from within `library`).
    subpattern_name, subpattern = library.mkpat(...)
    subpattern.rect(...)
    ...
    return subpattern_name
my_pattern.ref(_make_my_subpattern(), offset=..., ...)
```


## TODO

* Better interface for polygon operations (e.g. with `pyclipper`)
    - de-embedding
    - boolean ops
* Tests tests tests
* check renderpather
* pather and renderpather examples
