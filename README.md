# Masque README

Masque is a Python module for designing lithography masks.

The general idea is to implement something resembling the GDSII file-format, but
with some vectorized element types (eg. circles, not just polygons), better support for
E-beam doses, and the ability to output to multiple formats.

- [Source repository](https://mpxd.net/code/jan/masque)
- [PyPi](https://pypi.org/project/masque)


## Installation

Requirements:
* python >= 3.5 (written and tested with 3.6)
* numpy
* matplotlib (optional, used for visualization functions and text)
* python-gdsii (optional, used for gdsii i/o)
* svgwrite (optional, used for svg output)
* freetype (optional, used for text)


Install with pip:
```bash
pip3 install masque
```

Alternatively, install from git
```bash
pip3 install git+https://mpxd.net/code/jan/masque.git@release
```

## TODO

* Mirroring
* Polygon de-embedding

### Maybe

* Construct from bitmap
* Boolean operations on polygons (using pyclipper)
* Output to OASIS (using fatamorgana)
