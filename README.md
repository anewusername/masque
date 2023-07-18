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
* klamath (optional, used for `gdsii` i/o)
* matplotlib (optional, used for `visualization` functions and `text`)
* ezdxf (optional, used for `dxf` i/o)
* fatamorgana (optional, used for `oasis` i/o)
* svgwrite (optional, used for `svg` output)
* freetype (optional, used for `text`)


Install with pip:
```bash
pip install 'masque[visualization,oasis,dxf,svg,text]'
```

Alternatively, install from git
```bash
pip install git+https://mpxd.net/code/jan/masque.git@release
```

## Glossary
- `Library`: OASIS or GDS "library" or file (a collection of named cells)
- `Pattern`: OASIS or GDS "Cell", DXF "Block"
- `Ref`: GDS "AREF/SREF", OASIS "Placement"
- `Shape`: OASIS or GDS "Geometry element", DXF "LWPolyline" or "Polyline"
- `repetition`: OASIS "repetition". GDS "AREF" is a `Ref` combined with a `Grid` repetition.
- `Label`: OASIS, GDS, DXF "Text".
- `annotation`: OASIS or GDS "property"


## TODO

* Better interface for polygon operations (e.g. with `pyclipper`)
    - de-embedding
    - boolean ops
* DOCS DOCS DOCS
* Tests tests tests
* check renderpather
* pather and renderpather examples
