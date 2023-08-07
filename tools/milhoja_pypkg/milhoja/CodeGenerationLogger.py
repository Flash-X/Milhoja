import sys

class CodeGenerationLogger(object):
    """
    It is intended that all Milhoja code generation tools use this class to
    create loggers and that all log, warning, and error messages from those
    tools be written using the logger.  In this way, the Milhoja code generators
    will present a common, uniform logging interface to all users.

    Note that this includes the Milhoja code generation scripts to be called by
    applications.  For instance, these scripts should log errors through a
    logger rather than printing to stdout/stderr themselves.
    """
    NO_LOGGING_LEVEL  = 0
    BASIC_LOG_LEVEL   = 1
    BASIC_DEBUG_LEVEL = 2
    MAX_LOG_LEVEL     = 3

    LOG_LEVELS = list(range(NO_LOGGING_LEVEL, MAX_LOG_LEVEL+1))

    def __init__(self, tool_name, level):
        """
        Logger constructor.

        :param tool_name: The name of the tool that will use the logger.  This
            name will appear in all messages.
        :type  tool_name: str
        :param level: The verbosity logging level of the logger.
        :type  level: int contained in CodeGenerationLogger.LOG_LEVELS
        """
        super().__init__()

        self.__tool_name = tool_name

        self.__level = level
        if self.__level not in CodeGenerationLogger.LOG_LEVELS:
            # Calling code do not have access to the logger for printing the
            # error message.  Print it on their behalf.
            msg = f"Invalid code generation logging level ({self.__level})"
            self.error(msg)
            raise ValueError(msg)

    @property
    def level(self):
        """
        All log messages with a level greater than or equal to this value will
        be logged.  Warning and errors are logged regardless of this value.
        """
        return self.__level

    def log(self, msg, min_level):
        """
        Print the given message to stdout if the logger's verbosity level is
        greater than or equal to the given logging threshold level.

        :param msg: The message to potentially log
        :type  msg: str
        :param min_level: The threshold logging level
        :type  min_level: int contained in CodeGenerationLogger.LOG_LEVELS
        """
        valid = set(CodeGenerationLogger.LOG_LEVELS)
        valid = valid.difference(set([CodeGenerationLogger.NO_LOGGING_LEVEL]))
        if min_level not in valid:
            raise ValueError(f"Invalid code generation logging level ({min_level})")

        if self.__level >= min_level:
            sys.stdout.write(f"[{self.__tool_name}] {msg}\n")
            sys.stdout.flush()

    def warn(self, msg):
        """
        Print the given message to stdout in such a way that it is clear that it
        is transmitting a warning message to users.  This is printed regardless
        of the logger's verbosity level.

        :param msg: The warning message to log
        :type  msg: str
        """
        # ANSI terminal colors
        FAILURE  = '\033[0;91;1m' # Bright Red/bold
        NC       = '\033[0m'      # No Color/Not bold

        sys.stdout.write(f"[{self.__tool_name}] {FAILURE}WARNING{NC} - {msg}\n")
        sys.stdout.flush()

    def error(self, msg):
        """
        Print the given message to stderr in such a way that it is clear that it
        is transmitting an error message to users.  This is printed regardless
        of the logger's verbosity level.

        :param msg: The error message to log
        :type  msg: str
        """
        # ANSI terminal colors
        FAILURE  = '\033[0;91;1m' # Bright Red/bold
        NC       = '\033[0m'      # No Color/Not bold

        sys.stderr.write(f"[{self.__tool_name}] {FAILURE}ERROR - {msg}{NC}\n")
        sys.stderr.flush()

