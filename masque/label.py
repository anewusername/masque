from typing import Self
import copy

import numpy
from numpy.typing import ArrayLike, NDArray

from .repetition import Repetition
from .utils import rotation_matrix_2d, annotations_t
from .traits import PositionableImpl, Copyable, Pivotable, RepeatableImpl, Bounded
from .traits import AnnotatableImpl


class Label(PositionableImpl, RepeatableImpl, AnnotatableImpl, Bounded, Pivotable, Copyable):
    """
    A text annotation with a position (but no size; it is not drawn)
    """
    __slots__ = (
        '_string',
        # Inherited
        '_offset', '_repetition', '_annotations',
        )

    _string: str
    """ Label string """

    '''
    ---- Properties
    '''
    # string property
    @property
    def string(self) -> str:
        """
        Label string (str)
        """
        return self._string

    @string.setter
    def string(self, val: str) -> None:
        self._string = val

    def __init__(
            self,
            string: str,
            *,
            offset: ArrayLike = (0.0, 0.0),
            repetition: Repetition | None = None,
            annotations: annotations_t | None = None,
            ) -> None:
        self.string = string
        self.offset = numpy.array(offset, dtype=float, copy=True)
        self.repetition = repetition
        self.annotations = annotations if annotations is not None else {}

    def __copy__(self) -> Self:
        return type(self)(
            string=self.string,
            offset=self.offset.copy(),
            repetition=self.repetition,
            )

    def __deepcopy__(self, memo: dict | None = None) -> Self:
        memo = {} if memo is None else memo
        new = copy.copy(self)
        new._offset = self._offset.copy()
        return new

    def rotate_around(self, pivot: ArrayLike, rotation: float) -> Self:
        """
        Rotate the label around a point.

        Args:
            pivot: Point (x, y) to rotate around
            rotation: Angle to rotate by (counterclockwise, radians)

        Returns:
            self
        """
        pivot = numpy.array(pivot, dtype=float)
        self.translate(-pivot)
        self.offset = numpy.dot(rotation_matrix_2d(rotation), self.offset)
        self.translate(+pivot)
        return self

    def get_bounds_single(self) -> NDArray[numpy.float64]:
        """
        Return the bounds of the label.

        Labels are assumed to take up 0 area, i.e.
        bounds = [self.offset,
                  self.offset]

        Returns:
            Bounds [[xmin, xmax], [ymin, ymax]]
        """
        return numpy.array([self.offset, self.offset])
