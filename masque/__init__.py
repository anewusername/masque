"""
 masque 2D CAD library

 masque is an attempt to make a relatively small library for designing lithography
  masks. The general idea is to implement something resembling the GDSII and OASIS file-formats,
  but with some additional vectorized element types (eg. ellipses, not just polygons), better
  support for E-beam doses, and the ability to interface with multiple file formats.

 `Pattern` is a basic object containing a 2D lithography mask, composed of a list of `Shape`
  objects, a list of `Label` objects, and a list of references to other `Patterns` (using
  `SubPattern`).

 `SubPattern` provides basic support for nesting `Pattern` objects within each other, by adding
  offset, rotation, scaling, repetition, and other such properties to a Pattern reference.

 Note that the methods for these classes try to avoid copying wherever possible, so unless
  otherwise noted, assume that arguments are stored by-reference.


 NOTES ON INTERNALS
 ==========================
 - Many of `masque`'s classes make use of `__slots__` to make them faster / smaller.
    Since `__slots__` doesn't play well with multiple inheritance, the `masque.utils.AutoSlots`
    metaclass is used to auto-generate slots based on superclass type annotations.
 - File I/O submodules are imported by `masque.file` to avoid creating hard dependencies on
    external file-format reader/writers
 - Pattern locking/unlocking is quite slow for large hierarchies.

"""

from .error import PatternError, PatternLockedError
from .shapes import Shape
from .label import Label
from .subpattern import SubPattern
from .pattern import Pattern
from .utils import layer_t, annotations_t
from .library import Library, DeviceLibrary


__author__ = 'Jan Petykiewicz'

from .VERSION import __version__
version = __version__       # legacy
