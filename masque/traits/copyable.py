from typing import Self
import copy


class Copyable:
    """
    Trait class which adds .copy() and .deepcopy()
    """
    __slots__ = ()

    #
    # Non-abstract methods
    #
    def copy(self) -> Self:
        """
        Return a shallow copy of the object.

        Returns:
            `copy.copy(self)`
        """
        return copy.copy(self)

    def deepcopy(self) -> Self:
        """
        Return a deep copy of the object.

        Returns:
            `copy.deepcopy(self)`
        """
        return copy.deepcopy(self)
