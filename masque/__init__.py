"""
 masque 2D CAD library

 masque is an attempt to make a relatively small library for designing lithography
  masks. The general idea is to implement something resembling the GDSII file-format, but
  with some vectorized element types (eg. circles, not just polygons), better support for
  E-beam doses, and the ability to output to multiple formats.

 `Pattern` is a basic object containing a 2D lithography mask, composed of a list of `Shape`
  objects, a list of `Label` objects, and a list of references to other `Patterns` (using
  `SubPattern`).

 `SubPattern` provides basic support for nesting `Pattern` objects within each other, by adding
  offset, rotation, scaling, repetition, and other such properties to a Pattern reference.

 Note that the methods for these classes try to avoid copying wherever possible, so unless
  otherwise noted, assume that arguments are stored by-reference.
"""

import pathlib

from .error import PatternError, PatternLockedError
from .shapes import Shape
from .label import Label
from .subpattern import SubPattern
from .pattern import Pattern
from .utils import layer_t, annotations_t
from .library import Library


__author__ = 'Jan Petykiewicz'

with open(pathlib.Path(__file__).parent / 'VERSION', 'r') as f:
    __version__ = f.read().strip()
version = __version__
