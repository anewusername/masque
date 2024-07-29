"""
Traits (mixins) and default implementations

Traits and mixins should set `__slots__ = ()` to enable use of `__slots__` in subclasses.
"""
from .positionable import (
    Positionable as Positionable,
    PositionableImpl as PositionableImpl,
    Bounded as Bounded,
    )
from .layerable import (
    Layerable as Layerable,
    LayerableImpl as LayerableImpl,
    )
from .rotatable import (
    Rotatable as Rotatable,
    RotatableImpl as RotatableImpl,
    Pivotable as Pivotable,
    PivotableImpl as PivotableImpl,
    )
from .repeatable import (
    Repeatable as Repeatable,
    RepeatableImpl as RepeatableImpl,
    )
from .scalable import (
    Scalable as Scalable,
    ScalableImpl as ScalableImpl,
    )
from .mirrorable import Mirrorable as Mirrorable
from .copyable import Copyable as Copyable
from .annotatable import (
    Annotatable as Annotatable,
    AnnotatableImpl as AnnotatableImpl,
    )
