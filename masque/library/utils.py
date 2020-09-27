from typing import Callable, TypeVar, Generic
from functools import lru_cache


Key = TypeVar('Key')
Value = TypeVar('Value')


class DeferredDict(dict, Generic[Key, Value]):
    """
    This is a modified `dict` which is used to defer loading/generating
     values until they are accessed.

    ```
    bignum = my_slow_function()         # slow function call, would like to defer this
    numbers = Library()
    numbers['big'] = my_slow_function        # no slow function call here
    assert(bignum == numbers['big'])    # first access is slow (function called)
    assert(bignum == numbers['big'])    # second access is fast (result is cached)
    ```

    The `set_const` method is provided for convenience;
     `numbers['a'] = lambda: 10` is equivalent to `numbers.set_const('a', 10)`.
    """
    def __init__(self, *args, **kwargs) -> None:
        dict.__init__(self)
        self.update(*args, **kwargs)

    def __setitem__(self, key: Key, value: Callable[[], Value]) -> None:
        cached_fn = lru_cache(maxsize=1)(value)
        dict.__setitem__(self, key, cached_fn)

    def __getitem__(self, key: Key) -> Value:
        return dict.__getitem__(self, key)()

    def update(self, *args, **kwargs) -> None:
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __repr__(self) -> str:
        return '<Library with keys ' + repr(set(self.keys())) + '>'

    def set_const(self, key: Key, value: Value) -> None:
        """
        Convenience function to avoid having to manually wrap
         constant values into callables.
        """
        self[key] = lambda: value
