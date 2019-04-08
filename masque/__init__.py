"""
 masque 2D CAD library

 masque is an attempt to make a relatively small library for designing lithography
  masks. The general idea is to implement something resembling the GDSII file-format, but
  with some vectorized element types (eg. circles, not just polygons), better support for
  E-beam doses, and the ability to output to multiple formats.

 Pattern is a basic object containing a 2D lithography mask, composed of a list of Shape
  objects and a list of SubPattern objects.

 SubPattern provides basic support for nesting Pattern objects within each other, by adding
  offset, rotation, scaling, and other such properties to a Pattern reference.

 Note that the methods for these classes try to avoid copying wherever possible, so unless
  otherwise noted, assume that arguments are stored by-reference.


 Dependencies:
    - numpy
    - matplotlib    [Pattern.visualize(...)]
    - python-gdsii  [masque.file.gdsii]
    - svgwrite      [masque.file.svg]
"""

from .error import PatternError
from .shapes import Shape
from .label import Label
from .subpattern import SubPattern
from .repetition import GridRepetition
from .pattern import Pattern


__author__ = 'Jan Petykiewicz'

version = '0.5'
