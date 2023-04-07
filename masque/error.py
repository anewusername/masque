class MasqueError(Exception):
    """
    Parent exception for all Masque-related Exceptions
    """
    pass


class PatternError(MasqueError):
    """
    Exception for Pattern objects and their contents
    """
    pass


class LibraryError(MasqueError):
    """
    Exception raised by Library classes
    """
    pass


class BuildError(MasqueError):
    """
    Exception raised by builder-related functions
    """
    pass

class PortError(MasqueError):
    """
    Exception raised by builder-related functions
    """
    pass

class OneShotError(MasqueError):
    """
    Exception raised when a function decorated with `@oneshot` is called more than once
    """
    def __init__(self, func_name: str) -> None:
        Exception.__init__(self, f'Function "{func_name}" with @oneshot was called more than once')
