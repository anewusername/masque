from abc import ABCMeta


class AutoSlots(ABCMeta):
    """
    Metaclass for automatically generating __slots__ based on superclass type annotations.

    Superclasses must set `__slots__ = ()` to make this work properly.

    This is a workaround for the fact that non-empty `__slots__` can't be used
    with multiple inheritance. Since we only use multiple inheritance with abstract
    classes, they can have empty `__slots__` and their attribute type annotations
    can be used to generate a full `__slots__` for the concrete class.
    """
    def __new__(cls, name, bases, dctn):
        parents = set()
        for base in bases:
            parents |= set(base.mro())

        slots = tuple(dctn.get('__slots__', tuple()))
        for parent in parents:
            if not hasattr(parent, '__annotations__'):
                continue
            slots += tuple(getattr(parent, '__annotations__').keys())

        dctn['__slots__'] = slots
        return super().__new__(cls, name, bases, dctn)
