from typing import TypeVar, Dict, Tuple, Any
from abc import ABCMeta, abstractmethod

from ..error import PatternLockedError


T = TypeVar('T', bound='Lockable')
I = TypeVar('I', bound='LockableImpl')


class Lockable(metaclass=ABCMeta):
    """
    Abstract class for all lockable entities
    """
    __slots__ = ()      # type: Tuple[str, ...]

    '''
    ---- Methods
    '''
    @abstractmethod
    def lock(self: T) -> T:
        """
        Lock the object, disallowing further changes

        Returns:
            self
        """
        pass

    @abstractmethod
    def unlock(self: T) -> T:
        """
        Unlock the object, reallowing changes

        Returns:
            self
        """
        pass

    @abstractmethod
    def is_locked(self) -> bool:
        """
        Returns:
            True if the object is locked
        """
        pass

    def set_locked(self: T, locked: bool) -> T:
        """
        Locks or unlocks based on the argument.
        No action if already in the requested state.

        Args:
            locked: State to set.

        Returns:
            self
        """
        if locked != self.is_locked():
            if locked:
                self.lock()
            else:
                self.unlock()
        return self


class LockableImpl(Lockable, metaclass=ABCMeta):
    """
    Simple implementation of Lockable
    """
    __slots__ = ()      # type: Tuple[str, ...]

    locked: bool
    """ If `True`, disallows changes to the object """

    '''
    ---- Non-abstract methods
    '''
    def __setattr__(self, name, value):
        if self.locked and name != 'locked':
            raise PatternLockedError()
        object.__setattr__(self, name, value)

    def __getstate__(self) -> Dict[str, Any]:
        if hasattr(self, '__slots__'):
            return {key: getattr(self, key) for key in self.__slots__}
        else:
            return self.__dict__

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, v)

    def lock(self: I) -> I:
        object.__setattr__(self, 'locked', True)
        return self

    def unlock(self: I) -> I:
        object.__setattr__(self, 'locked', False)
        return self

    def is_locked(self) -> bool:
        return self.locked
