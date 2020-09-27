class PatternError(Exception):
    """
    Simple Exception for Pattern objects and their contents
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class PatternLockedError(PatternError):
    """
    Exception raised when trying to modify a locked pattern
    """
    def __init__(self):
        PatternError.__init__(self, 'Tried to modify a locked Pattern, subpattern, or shape')


class LibraryError(Exception):
    """
    Exception raised by Library classes
    """
    pass


