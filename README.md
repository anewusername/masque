# Masque README

Masque is a Python module for designing lithography masks.

The general idea is to implement something resembling the GDSII file-format, but
with some vectorized element types (eg. circles, not just polygons), better support for
E-beam doses, and the ability to output to multiple formats.

- [Source repository](https://mpxd.net/code/jan/masque)
- [PyPI](https://pypi.org/project/masque)


## Installation

Requirements:
* python >= 3.7 (written and tested with 3.8)
* numpy
* matplotlib (optional, used for `visualization` functions and `text`)
* python-gdsii (optional, used for `gdsii` i/o)
* klamath (optional, used for `gdsii` i/o and library management)
* ezdxf (optional, used for `dxf` i/o)
* fatamorgana (optional, used for `oasis` i/o)
* svgwrite (optional, used for `svg` output)
* freetype (optional, used for `text`)


Install with pip:
```bash
pip3 install 'masque[visualization,gdsii,oasis,dxf,svg,text,klamath]'
```

Alternatively, install from git
```bash
pip3 install git+https://mpxd.net/code/jan/masque.git@release
```

## Translation
- `Pattern`: OASIS or GDS "Cell", DXF "Block"
- `SubPattern`: GDS "AREF/SREF", OASIS "Placement"
- `Shape`: OASIS or GDS "Geometry element", DXF "LWPolyline" or "Polyline"
- `repetition`: OASIS "repetition". GDS "AREF" is a `SubPattern` combined with a `Grid` repetition.
- `Label`: OASIS, GDS, DXF "Text".
- `annotation`: OASIS or GDS "property"


## TODO

* Better interface for polygon operations (e.g. with `pyclipper`)
    - de-embedding
    - boolean ops
* Construct polygons from bitmap using `skimage.find_contours`
* Deal with shape repetitions for dxf, svg
