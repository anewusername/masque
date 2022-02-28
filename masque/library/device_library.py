"""
DeviceLibrary class for managing unique name->device mappings and
 deferred loading or creation.
"""
from typing import Dict, Callable, TypeVar, TYPE_CHECKING
from typing import Any, Tuple, Union, Iterator
import logging
from pprint import pformat

from ..error import DeviceLibraryError

if TYPE_CHECKING:
    from ..builder import Device


logger = logging.getLogger(__name__)


L = TypeVar('L', bound='DeviceLibrary')


class DeviceLibrary:
    """
    This class maps names to functions which generate or load the
     relevant `Device` object.

    This class largely functions the same way as `Library`, but
     operates on `Device`s rather than `Patterns` and thus has no
     need for distinctions between primary/secondary devices (as
     there is no inter-`Device` hierarchy).

    Each device is cached the first time it is used. The cache can
     be disabled by setting the `enable_cache` attribute to `False`.
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

    def set_const(self, const: 'Device') -> None:
        """
        Convenience function to avoid having to manually wrap
         already-generated Device objects into callables.

        Args:
            const: Pre-generated device object
        """
        self.generators[const.pattern.name] = lambda: const

    def add(
            self: D,
            other: D,
            use_ours: Callable[[str], bool] = lambda name: False,
            use_theirs: Callable[[str], bool] = lambda name: False,
            ) -> D:
        """
        Add keys from another library into this one.

        There must be no conflicting keys.

        Args:
            other: The library to insert keys from
            use_ours: Decision function for name conflicts. Will be called with duplicate cell names.
                Should return `True` if the value from `self` should be used.
            use_theirs: Decision function for name conflicts. Same format as `use_ours`.
                Should return `True` if the value from `other` should be used.
                `use_ours` takes priority over `use_theirs`.

        Returns:
            self
        """
        duplicates = set(self.keys()) & set(other.keys())
        keep_ours = set(name for name in duplicates if use_ours(name))
        keep_theirs = set(name for name in duplicates - keep_ours if use_theirs(name))
        conflicts = duplicates - keep_ours - keep_theirs
        if conflicts:
            raise DeviceLibraryError('Duplicate keys encountered in DeviceLibrary merge: '
                                     + pformat(conflicts))

        for name in set(other.generators.keys()) - keep_ours:
            self.generators[name] = other.generators[name]
            if name in other.cache:
                self.cache[name] = other.cache[name]
        return self

    def clear_cache(self: D) -> D:
        """
        Clear the cache of this library.
        This is usually used before modifying or deleting cells, e.g. when merging
          with another library.

        Returns:
            self
        """
        self.cache = {}
        return self

    def add_device(
            self,
            name: str,
            fn: Callable[[], 'Device'],
            dev2pat: Callable[['Device'], 'Pattern'],
            prefix: str = '',
            ) -> None:
        """
        Convenience function for adding a device to the library.

        - The device is generated with the provided `fn()`
        - Port info is written to the pattern using the provied dev2pat
        - The pattern is renamed to match the provided `prefix + name`
        - If `prefix` is non-empty, a wrapped copy is also added, named
            `name` (no prefix). See `wrap_device()` for details.

        Adding devices with this function helps to
        - Make sure Pattern names are reflective of what the devices are named
        - Ensure port info is written into the `Pattern`, so that the `Device`
            can be reconstituted from the layout.
        - Simplify adding a prefix to all device names, to make it easier to
            track their provenance and purpose, while also allowing for
            generic device names which can later be swapped out with different
            underlying implementations.

        Args:
            name: Base name for the device. If a prefix is used, this is the
                "generic" name (e.g. "L3_cavity" vs "2022_02_02_L3_cavity").
            fn: Function which is called to generate the device.
            dev2pat: Post-processing function which is called to add the port
                info into the device's pattern.
            prefix: If present, the actual device is named `prefix + name`, and
                a second device with name `name` is also added (containing only
                this one).
        """
        def build_dev() -> 'Device':
            dev = fn()
            dev.pattern = dev2pat(dev)
            dev.pattern.rename(prefix + name)
            return dev

        self[prefix + name] = build_dev
        if prefix:
            self.wrap_device(name, prefix + name)

