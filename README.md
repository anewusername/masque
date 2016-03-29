# Masque README

Masque is a Python module for designing lithography masks.

The general idea is to implement something resembling the GDSII file-format, but
with some vectorized element types (eg. circles, not just polygons), better support for
E-beam doses, and the ability to output to multiple formats.

## Installation

Requirements:
* python 3 (written and tested with 3.5)
* numpy
* matplotlib (optional, used for visualization functions)
* python-gdsii (optional, used for gdsii i/o)
* svgwrite (optional, used for svg output)


Install with pip, via git:
```bash
pip install git+https://mpxd.net/gogs/jan/masque.git@release
```
