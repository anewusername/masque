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

from .utils import (
    layer_t as layer_t,
    annotations_t as annotations_t,
    SupportsBool as SupportsBool,
    )
from .error import (
    MasqueError as MasqueError,
    PatternError as PatternError,
    LibraryError as LibraryError,
    BuildError as BuildError,
    )
from .shapes import (
    Shape as Shape,
    Polygon as Polygon,
    Path as Path,
    Circle as Circle,
    Arc as Arc,
    Ellipse as Ellipse,
    )
from .label import Label as Label
from .ref import Ref as Ref
from .pattern import (
    Pattern as Pattern,
    map_layers as map_layers,
    map_targets as map_targets,
    chain_elements as chain_elements,
    )

from .library import (
    ILibraryView as ILibraryView,
    ILibrary as ILibrary,
    LibraryView as LibraryView,
    Library as Library,
    LazyLibrary as LazyLibrary,
    AbstractView as AbstractView,
    TreeView as TreeView,
    Tree as Tree,
    )
from .ports import (
    Port as Port,
    PortList as PortList,
    )
from .abstract import Abstract as Abstract
from .builder import (
    Builder as Builder,
    Tool as Tool,
    Pather as Pather,
    RenderPather as RenderPather,
    RenderStep as RenderStep,
    BasicTool as BasicTool,
    PathTool as PathTool,
    )
from .utils import (
    ports2data as ports2data,
    oneshot as oneshot,
    R90 as R90,
    R180 as R180,
    )


__author__ = 'Jan Petykiewicz'

__version__ = '3.3'
version = __version__       # legacy
