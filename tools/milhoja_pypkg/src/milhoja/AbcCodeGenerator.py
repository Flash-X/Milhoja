import abc

from . import AbcLogger
from . import TaskFunction


class AbcCodeGenerator(abc.ABC):
    """
    This class takes the TaskFunction specification object as an argument
    rather than instantiate one from arguments so that application codes can
    use a custom class derived from TaskFunction if so desired.  For the same
    reason, code generators derived from this class should also take the
    specification as an object.

    This class also takes a logger object instantiated from a concrete class
    derived from AbcLogger so that application codes can use custom logging if
    so desired.  For the same reason, code generators derived from this class
    should also take the logger as an object.

    .. todo::
        * Make ``_logger`` private once all code generators are derived from
          this class such that they can use the logging interface of this
          class.
    """
    def __init__(
                self,
                tf_specification: TaskFunction,
                header_filename,
                source_filename,
                indent,
                log_tag, logger
            ):
        """
        """
        super().__init__()

        # Allow derived class to access specification object directly
        self._tf_spec = tf_specification

        self.__indent = indent

        # These should just be filenames with no path.  Concrete classes will
        # be instructed at generation where the files should be written.
        self.__hdr_filename = header_filename
        self.__src_filename = source_filename

        # Allow derived classes access to logger in case they need to pass it
        # to functions external to the class.
        self._logger = logger
        self.__log_tag = log_tag

        # ----- SANITY CHECK ARGUMENTS
        # Since there could be no header or source files at instantiation, but
        # a file could appear before calling a generate method, we don't check
        # file existence here.  Rather, concrete derived classes should check
        # for pre-existing files.
        if not isinstance(self._tf_spec, TaskFunction):
            raise TypeError("Given tf_spec not derived from TaskFunction")
        if self.__indent < 0:
            raise ValueError(f"Negative code generation indent ({indent})")
        if not isinstance(self._logger, AbcLogger):
            raise TypeError("Logger not derived from milhoja.AbcLogger")

    @property
    def specification_filename(self):
        """
        """
        return self._tf_spec.specification_filename

    @property
    def indentation(self):
        """
        """
        return self.__indent

    @property
    def verbosity_level(self):
        """
        """
        return self._logger.level

    def _log(self, msg, level):
        """
        """
        self._logger.log(self.__log_tag, msg, level)

    def _warn(self, msg):
        """
        """
        self._logger.warn(self.__log_tag, msg)

    def _error(self, msg):
        """
        """
        self._logger.error(self.__log_tag, msg)

    @property
    def header_filename(self):
        """
        """
        return self.__hdr_filename

    @property
    def source_filename(self):
        """
        """
        return self.__src_filename

    @abc.abstractmethod
    def generate_header_code(self, destination, overwrite):
        """
        """
        ...

    @abc.abstractmethod
    def generate_source_code(self, destination, overwrite):
        """
        """
        ...
