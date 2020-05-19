# Masque README

Masque is a Python module for designing lithography masks.

The general idea is to implement something resembling the GDSII file-format, but
with some vectorized element types (eg. circles, not just polygons), better support for
E-beam doses, and the ability to output to multiple formats.

- [Source repository](https://mpxd.net/code/jan/masque)
- [PyPi](https://pypi.org/project/masque)


## Installation

Requirements:
* python >= 3.7 (written and tested with 3.8)
* numpy
* matplotlib (optional, used for `visualization` functions and `text`)
* python-gdsii (optional, used for `gdsii` i/o)
* ezdxf (optional, used for `dxf` i/o)
* fatamorgana (optional, used for `oasis` i/o)
* svgwrite (optional, used for `svg` output)
* freetype (optional, used for `text`)


Install with pip:
```bash
pip3 install 'masque[visualization,gdsii,oasis,dxf,svg,text]'
```

Alternatively, install from git
```bash
pip3 install git+https://mpxd.net/code/jan/masque.git@release
```

## TODO

* Polygon de-embedding
* Construct from bitmap
* Boolean operations on polygons (using pyclipper)
* Implement shape/cell properties
* Implement OASIS-style repetitions for shapes
