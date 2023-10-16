"""
 masque 2D CAD library

 masque is an attempt to make a relatively compact library for designing lithography
  masks. The general idea is to implement something resembling the GDSII and OASIS file-formats,
  but with some additional vectorized element types (eg. ellipses, not just polygons), and the
  ability to interface with multiple file formats.

 `Pattern` is a basic object containing a 2D lithography mask, composed of a list of `Shape`
  objects, a list of `Label` objects, and a list of references to other `Patterns` (using
  `Ref`).

 `Ref` provides basic support for nesting `Pattern` objects within each other, by adding
  offset, rotation, scaling, repetition, and other such properties to a Pattern reference.

 Note that the methods for these classes try to avoid copying wherever possible, so unless
  otherwise noted, assume that arguments are stored by-reference.


 NOTES ON INTERNALS
 ==========================
 - Many of `masque`'s classes make use of `__slots__` to make them faster / smaller.
    Since `__slots__` doesn't play well with multiple inheritance, often they are left
    empty for superclasses and it is the subclass's responsibility to set them correctly.
 - File I/O submodules are not imported by `masque.file` to avoid creating hard dependencies
    on external file-format reader/writers
- Try to accept the broadest-possible inputs: e.g., don't demand an `ILibraryView` if you
    can accept a `Mapping[str, Pattern]` and wrap it in a `LibraryView` internally.
"""

from .utils import layer_t, annotations_t, SupportsBool
from .error import MasqueError, PatternError, LibraryError, BuildError
from .shapes import Shape, Polygon, Path, Circle, Arc, Ellipse
from .label import Label
from .ref import Ref
from .pattern import Pattern, map_layers, map_targets, chain_elements

from .library import (
    ILibraryView, ILibrary,
    LibraryView, Library, LazyLibrary,
    AbstractView,
    )
from .ports import Port, PortList
from .abstract import Abstract
from .builder import Builder, Tool, Pather, RenderPather, RenderStep, BasicTool, PathTool
from .utils import ports2data, oneshot


__author__ = 'Jan Petykiewicz'

__version__ = '3.0'
version = __version__       # legacy
