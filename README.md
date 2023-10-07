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
Each `Pattern` can contain `Ref`s pointing at other patterns, `Shape`s, and `Label`s.

`masque` departs from several "classic" GDSII paradigms:
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


## Glossary
- `Library`: A collection of named cells. OASIS or GDS "library" or file.
- `Pattern`: A collection of geometry, text labels, and reference to other patterns.
        OASIS or GDS "Cell", DXF "Block".
- `Ref`: A reference to another pattern. GDS "AREF/SREF", OASIS "Placement".
- `Shape`: Individual geometric entity. OASIS or GDS "Geometry element", DXF "LWPolyline" or "Polyline".
- `repetition`: Repetition operation. OASIS "repetition".
        GDS "AREF" is a `Ref` combined with a `Grid` repetition.
- `Label`: Text label. Not rendered into geometry. OASIS, GDS, DXF "Text".
- `annotation`: Additional metadata. OASIS or GDS "property".



## TODO

* Better interface for polygon operations (e.g. with `pyclipper`)
    - de-embedding
    - boolean ops
* Tests tests tests
* check renderpather
* pather and renderpather examples
