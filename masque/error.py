import traceback
import pathlib


MASQUE_DIR = str(pathlib.Path(__file__).parent)


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
    Exception raised by port-related functions
    """
    pass


class OneShotError(MasqueError):
    """
    Exception raised when a function decorated with `@oneshot` is called more than once
    """
    def __init__(self, func_name: str) -> None:
        Exception.__init__(self, f'Function "{func_name}" with @oneshot was called more than once')


def format_stacktrace(
        stacklevel: int = 1,
        *,
        skip_file_prefixes: tuple[str, ...] = (MASQUE_DIR,),
        low_file_prefixes: tuple[str, ...] = ('<frozen', '<runpy', '<string>'),
        low_file_suffixes: tuple[str, ...] = ('IPython/utils/py3compat.py', 'concurrent/futures/process.py'),
        ) -> str:
    """
    Utility function for making nicer stack traces (e.g. excluding <frozen runpy> and similar)

    Args:
        stacklevel: Number of frames to remove from near this function (default is to
            show caller but not ourselves). Similar to `warnings.warn` and `logging.warning`.
        skip_file_prefixes: Indicates frames to ignore after counting stack levels; similar
            to `warnings.warn` *TODO check if this is actually the same effect re:stacklevel*.
            Forces stacklevel to max(2, stacklevel).
            Default is to exclude anything within `masque`.
        low_file_prefixes: Indicates frames to ignore on the other (entry-point) end of the stack,
            based on prefixes on their filenames.
        low_file_suffixes: Indicates frames to ignore on the other (entry-point) end of the stack,
            based on suffixes on their filenames.

    Returns:
        Formatted trimmed stack trace
    """
    if skip_file_prefixes:
        stacklevel = max(2, stacklevel)

    stack = traceback.extract_stack()

    bad_inds = [ii + 1 for ii, frame in enumerate(stack)
                if frame.filename.startswith(low_file_prefixes) or frame.filename.endswith(low_file_suffixes)]
    first_ok = max([0] + bad_inds)

    last_ok = -stacklevel - 1
    while last_ok >= -len(stack) and stack[last_ok].filename.startswith(skip_file_prefixes):
        last_ok -= 1

    if selected := stack[first_ok:last_ok + 1]:
        pass
    elif selected := stack[:-stacklevel]:
        pass                      # noqa: SIM114     # separate elif for clarity
    else:
        selected = stack
    return ''.join(traceback.format_list(selected))
