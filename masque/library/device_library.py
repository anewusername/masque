"""
DeviceLibrary class for managing unique name->device mappings and
 deferred loading or creation.
"""
from typing import Dict, Callable, TypeVar, TYPE_CHECKING
from typing import Any, Tuple, Union, Iterator
import logging
from pprint import pformat

from ..error import LibraryError

if TYPE_CHECKING:
    from ..builder import Device


logger = logging.getLogger(__name__)


L = TypeVar('L', bound='DeviceLibrary')


class DeviceLibrary:
    """
    This class is usually used to create a device library by mapping names to
     functions which generate or load the relevant `Device` object as-needed.

    The cache can be disabled by setting the `enable_cache` attribute to `False`.
    """
    generators: Dict[str, Callable[[], 'Device']]
    cache: Dict[Union[str, Tuple[str, str]], 'Device']
    enable_cache: bool = True

    def __init__(self) -> None:
        self.generators = {}
        self.cache = {}

    def __setitem__(self, key: str, value: Callable[[], 'Device']) -> None:
        self.generators[key] = value
        if key in self.cache:
            del self.cache[key]

    def __delitem__(self, key: str) -> None:
        del self.generators[key]
        if key in self.cache:
            del self.cache[key]

    def __getitem__(self, key: str) -> 'Device':
        if self.enable_cache and key in self.cache:
            logger.debug(f'found {key} in cache')
            return self.cache[key]

        logger.debug(f'loading {key}')
        dev = self.generators[key]()
        self.cache[key] = dev
        return dev

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.generators

    def keys(self) -> Iterator[str]:
        return iter(self.generators.keys())

    def values(self) -> Iterator['Device']:
        return iter(self[key] for key in self.keys())

    def items(self) -> Iterator[Tuple[str, 'Device']]:
        return iter((key, self[key]) for key in self.keys())

    def __repr__(self) -> str:
        return '<DeviceLibrary with keys ' + repr(list(self.generators.keys())) + '>'

    def set_const(self, key: str, const: 'Device') -> None:
        """
        Convenience function to avoid having to manually wrap
         constant values into callables.

        Args:
            key: Lookup key, usually the device name
            const: Device object to return
        """
        self.generators[key] = lambda: const

    def add(self: L, other: L) -> L:
        """
        Add keys from another library into this one.

        There must be no conflicting keys.

        Args:
            other: The library to insert keys from

        Returns:
            self
        """
        conflicts = [key for key in other.generators
                     if key in self.generators]
        if conflicts:
            raise LibraryError('Duplicate keys encountered in library merge: ' + pformat(conflicts))

        self.generators.update(other.generators)
        self.cache.update(other.cache)
        return self
