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

class PatternLockedError(PatternError):
    """
    Exception raised when trying to modify a locked pattern
    """
    def __init__(self):
        PatternError.__init__(self, 'Tried to modify a locked Pattern, subpattern, or shape')


class LibraryError(MasqueError):
    """
    Exception raised by Library classes
    """
    pass


class DeviceLibraryError(MasqueError):
    """
    Exception raised by DeviceLibrary classes
    """
    pass


class DeviceError(MasqueError):
    """
    Exception raised by Device and Port objects
    """
    pass


class BuildError(MasqueError):
    """
    Exception raised by builder-related functions
    """
    pass
