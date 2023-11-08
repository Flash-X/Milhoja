import copy
import json

from pathlib import Path

from .constants import (
    MILHOJA_JSON_FORMAT, CURRENT_MILHOJA_JSON_VERSION,
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT
)
from .AbcLogger import AbcLogger
from .check_group_specification import check_group_specification


class SubroutineGroup(object):
    """
    Class that provides read-only access to the contents of a subroutine group
    specification.  In particular, calling code can alter any specification
    results obtained from a SubroutineGroup object without altering the
    original specification.
    """
    __LOG_TAG = "Milhoja Subroutine Group"

    @staticmethod
    def from_milhoja_json(filename, logger):
        """
        The given group specification is error checked completely so that
        results obtained from the returned object can be assumed to be correct.

        :param filename: Filename and path to Milhoja-JSON format subroutine
            group specification file.  This function does not alter the file.
        :param logger: Logger derived from :py:class:`milhoja.AbcLogger`

        :return: SubroutineGroup object that provides acccess to the
            specification provided in the given file
        """
        # ----- ERROR CHECK ARGUMENTS
        if (not isinstance(filename, str)) and (not isinstance(filename, Path)):
            raise TypeError(f"filename is not str or Path ({filename})")

        fname = Path(filename).resolve()
        if not fname.is_file():
            raise ValueError(f"{fname} does not exist or is not a file")
        elif not isinstance(logger, AbcLogger):
            raise TypeError("Unknown logger type")

        # ----- LOAD SPECIFICATION
        msg = f"Loading Milhoja-JSON subroutine group specification {fname}"
        logger.log(SubroutineGroup.__LOG_TAG, msg, LOG_LEVEL_BASIC)

        with open(filename, "r") as fptr:
            group_spec = json.load(fptr)

        # ----- CONVERT TO INTERNAL MILHOJA REPRESENTATION
        format_name, version = group_spec["format"]

        # Only one Milhoja-JSON format presently
        if format_name.lower() != MILHOJA_JSON_FORMAT.lower():
            raise ValueError(f"Unknown JSON format {format_name}")

        # Only one version of Milhoja-native JSON format.
        # Therefore, contents already in Milhoja-internal format.
        if version.lower() != CURRENT_MILHOJA_JSON_VERSION.lower():
            raise ValueError(f"Unknown {format_name} version v{version}")

        return SubroutineGroup(group_spec, logger)

    def __init__(self, group_spec, logger):
        """
        It is intended that calling code instantiate groups using the from_*
        classmethods rather than use the constructor.

        This checks the full specification of the group so that all other code
        can assume correct content rather than check themselves.

        :param group_spec:  A subroutine group specification dict provided
            in milhoja-internal format.  A deep copy of this is made
            immediately for internal use.  Therefore, neither this class nor
            code using this class can alter the contents of the actual argument
            passed in.
        :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
        """
        # ----- ERROR CHECK LOGGER & SETUP FOR IMMEDIATE USE
        if not isinstance(logger, AbcLogger):
            raise TypeError("Unknown logger type")
        self.__logger = logger

        # ----- ERROR CHECK SPECIFICATION
        self.__log_debug("Checking specification")
        self.__spec = copy.deepcopy(group_spec)
        check_group_specification(self.__spec, self.__logger)

        # ----- EAGERLY DETERMINE SUBROUTINES IN GROUP
        ignore = {"name", "variable_index_base",
                  EXTERNAL_ARGUMENT, SCRATCH_ARGUMENT}
        group_spec = self.__spec["operation"]
        self.__subroutines = set(group_spec).difference(ignore)
        assert self.__subroutines

        self.__log(f"Loaded {self.name} group")

    def __contains__(self, subroutine):
        """
        :return: True if given subroutine is included in group
        """
        if not isinstance(subroutine, str):
            msg = f"subroutine is not a string ({subroutine})"
            raise TypeError(msg)

        return subroutine in self.__subroutines

    def __log(self, msg):
        """
        Log given message at default log level
        """
        self.__logger.log(SubroutineGroup.__LOG_TAG, msg, LOG_LEVEL_BASIC)

    def __log_debug(self, msg):
        """
        Log given message at lowest debug log level
        """
        self.__logger.log(SubroutineGroup.__LOG_TAG, msg, LOG_LEVEL_BASIC_DEBUG)

    def __check_subroutine_arg(self, subroutine):
        """
        Raises an error if argument is not valid or not in the group
        """
        if not isinstance(subroutine, str):
            msg = f"Subroutine ({subroutine}) not a string"
            raise TypeError(msg)
        elif subroutine not in self.__subroutines:
            msg = "Subroutine ({}) not specified in group {}"
            raise ValueError(msg.format(subroutine, self.name))

    @property
    def name(self):
        """
        :return: Name of subroutine group
        """
        return self.__spec["operation"]["name"]

    @property
    def variable_index_base(self):
        """
        :return: All variable indices provided in R/RW/W fields in all
            subroutine specifications in the group must be elements of a single
            index set whose smallest index is this value.
        """
        return self.__spec["operation"]["variable_index_base"]

    @property
    def subroutines(self):
        """
        :return: Set of subroutines in the group
        """
        return self.__subroutines.copy()

    @property
    def specification(self):
        """
        :return: Milhoja-internal format of subroutine group specification as
            dict.  Calling code can alter the obtained data structure without
            affecting the contents of the SubroutineGroup object.
        """
        return copy.deepcopy(self.__spec)

    @property
    def group_external_variables(self):
        """
        :return: Set of group-level external variables
        """
        return set(self.__spec["operation"][EXTERNAL_ARGUMENT])

    @property
    def group_scratch_variables(self):
        """
        :return: Set of group-level scratch variables
        """
        return set(self.__spec["operation"][SCRATCH_ARGUMENT])

    def external_specification(self, external):
        """
        :param external: Name of group-level external variable
        :return: Specification of given variable
        """
        if not isinstance(external, str):
            msg = f"External variable name ({external}) not a string"
            raise TypeError(msg)
        elif external not in self.group_external_variables:
            msg = "({}) not group-level external variable in {}"
            raise ValueError(msg.format(external, self.name))

        spec = self.__spec["operation"][EXTERNAL_ARGUMENT][external]
        return copy.deepcopy(spec)

    def scratch_specification(self, scratch):
        """
        :param scratch: Name of group-level scratch variable
        :return: Specification of given variable
        """
        if not isinstance(scratch, str):
            msg = f"Scratch variable name ({scratch}) not a string"
            raise TypeError(msg)
        elif scratch not in self.group_scratch_variables:
            msg = "({}) not group-level scratch variable in {}"
            raise ValueError(msg.format(scratch, self.name))

        spec = self.__spec["operation"][SCRATCH_ARGUMENT][scratch]
        return copy.deepcopy(spec)

    def subroutine_specification(self, subroutine):
        """
        :return: Specification of given subroutine.  Calling code can alter the
            retured specification without affecting the contents of the
            SubroutineGroup object.
        """
        self.__check_subroutine_arg(subroutine)
        return copy.deepcopy(self.__spec["operation"][subroutine])

    def argument_list(self, subroutine):
        """
        :return: Argument list of given subroutine in correct order.  Calling
            code can alter the returned list without affecting the contents of
            the SubroutineGroup object.
        """
        self.__check_subroutine_arg(subroutine)
        sub_spec = self.__spec["operation"][subroutine]
        return sub_spec["argument_list"].copy()

    def argument_specification(self, subroutine, argument):
        """
        :param subroutine: Name of subroutine in group
        :param argument: Name of dummy argument for given subroutine

        :return: Specification of given argument.  Calling code can alter the
            returned specification without affecting the contents of the
            SubroutineGroup object.
        """
        if not isinstance(argument, str):
            msg = f"Argument ({argument}) not a string"
            raise TypeError(msg)

        sub_spec = self.subroutine_specification(subroutine)
        arg_specs_all = sub_spec["argument_specifications"]
        if argument not in arg_specs_all:
            msg = "Subroutine ({}) in group {} has no argument ({})"
            raise ValueError(msg.format(subroutine, self.name, argument))

        return copy.deepcopy(arg_specs_all[argument])
