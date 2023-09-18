"""
Traits (mixins) and default implementations

Traits and mixins should set `__slots__ = ()` to enable use of `__slots__` in subclasses.
"""
from .positionable import Positionable, PositionableImpl, Bounded
from .layerable import Layerable, LayerableImpl
from .rotatable import Rotatable, RotatableImpl, Pivotable, PivotableImpl
from .repeatable import Repeatable, RepeatableImpl
from .scalable import Scalable, ScalableImpl
from .mirrorable import Mirrorable
from .copyable import Copyable
from .annotatable import Annotatable, AnnotatableImpl
