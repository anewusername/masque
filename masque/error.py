class PatternError(Exception):
    """
    Simple Exception for Pattern objects and their contents
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
