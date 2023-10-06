import sys

from . import LOG_LEVELS
from . import LOG_LEVEL_NONE
from . import AbcLogger


class BasicLogger(AbcLogger):
    """
    A concrete Milhoja logger class that is written for use by the Milhoja Test
    sub-system and command line tools.

    .. note::
        Applications can use this logger so long as they understand that the
        implementation of this class might change as milhoja's testing and
        tooling needs change.
    """
    # ANSI terminal colors
    __FAILURE_COLOR = '\033[0;91;1m'  # Bright Red/bold
    __NO_COLOR = '\033[0m'            # No Color/Not bold

    def __init__(self, level):
        """
        :param level: Verbosity level of the logger
        """
        super().__init__(level)

    def log(self, caller, msg, min_level):
        """
        Print the given message to stdout if the logger's verbosity level is
        greater than or equal to the given logging threshold level.

        :param caller: Name of calling code for inclusion in actual logged
            message
        :param msg: Message to potentially log
        :param min_level: Message's log level
        """
        valid = set(LOG_LEVELS).difference(set([LOG_LEVEL_NONE]))
        if min_level not in valid:
            msg = f"Invalid code generation logging level ({min_level})"
            raise ValueError(msg)

        if self.level >= min_level:
            sys.stdout.write(f"[{caller}] {msg}\n")
            sys.stdout.flush()

    def warn(self, caller, msg):
        """
        Print the given message to stdout in such a way that it is clear that
        it is transmitting a warning message to users.  This is printed
        regardless of the logger's verbosity level.

        :param caller: Name of calling code for inclusion in actual logged
            warning
        :param msg: Warning message to log
        """
        sys.stdout.write("[{}] {}WARNING{} - {}\n".format(
            caller, BasicLogger.__FAILURE_COLOR, BasicLogger.__NO_COLOR, msg
        ))
        sys.stdout.flush()

    def error(self, caller, msg):
        """
        Print the given message to stderr in such a way that it is clear that
        it is transmitting an error message to users.  This is printed
        regardless of the logger's verbosity level.

        :param caller: Name of calling code for inclusion in actual logged
            error
        :param msg: Error message to log
        """
        sys.stderr.write("[{}] {}ERROR - {}{}\n".format(
            caller, BasicLogger.__FAILURE_COLOR, msg, BasicLogger.__NO_COLOR
        ))
        sys.stderr.write(msg)
        sys.stderr.flush()
