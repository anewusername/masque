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
