from typing import Callable
from functools import wraps

from ..error import OneShotError


def oneshot(func: Callable) -> Callable:
    """
    Raises a OneShotError if the decorated function is called more than once
    """
    expired = False

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal expired
        if expired:
            raise OneShotError(func.__name__)
        expired = True
        return func(*args, **kwargs)

    return wrapper
