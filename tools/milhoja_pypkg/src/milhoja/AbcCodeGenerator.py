import abc

from pathlib import Path

from . import CodeGenerationLogger

class AbcCodeGenerator(abc.ABC):
    """
    This class takes the task function specification as an argument rather than
    instantiate one from arguments so that application codes can use a custom
    class derived from TaskFunction if so desired.  For the same reason, code
    generators derived from this class should also take the specification as an
    object.
    """
    def __init__(
            self,
            tf_specification,
            header_filename,
            source_filename,
            log_tag, log_level,
            indent
        ):
        """
        """
        super().__init__()

        # Allow derived class to access specification object directly
        # TODO: Get rid of _new once the code generators use TaskFunction
        # completely.
        self._tf_spec_new = tf_specification

        self.__indent = indent

        # These should just be filenames with no path.  Concrete classes will
        # be instructed at generation where the files should be written.
        self.__hdr_filename = header_filename
        self.__src_filename = source_filename

        self.__logger = CodeGenerationLogger(log_tag, log_level)

        # ----- SANITY CHECK ARGUMENTS
        # Since there could be no header or source files at instantiation, but
        # a file could appear before calling a generate method, we don't check
        # file existence here.  Rather, concrete derived class' should check
        # for pre-existing files.
        if self.__indent < 0:
            raise ValueError(f"Negative code generation indent ({indent})")

    @property
    def specification_filename(self):
        """
        """
        return self._tf_spec_new.specification_filename

    @property
    def indentation(self):
        """
        """
        return self.__indent

    @property
    def verbosity_level(self):
        """
        """
        return self.__logger.level

    def _log(self, msg, level):
        """
        """
        self.__logger.log(msg, level)

    def _warn(self, msg):
        """
        """
        self.__logger.warn(msg)

    def _error(self, msg):
        """
        """
        self.__logger.error(msg)

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
