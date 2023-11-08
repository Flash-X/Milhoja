import copy
import json

import itertools as it

from pathlib import Path

from .constants import (
    MILHOJA_JSON_FORMAT, CURRENT_MILHOJA_JSON_VERSION,
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    EXTERNAL_ARGUMENT,
    SCRATCH_ARGUMENT,
    GRID_DATA_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT,
    TILE_ARGUMENTS_ALL,
    THREAD_INDEX_ARGUMENT, THREAD_INDEX_VAR_NAME
)
from .LogicError import LogicError
from .AbcLogger import AbcLogger
from .SubroutineGroup import SubroutineGroup
from .check_grid_specification import check_grid_specification


class TaskFunctionAssembler(object):
    """
    A class for assembling a single task function from application-specified
    information and writing its full specification to file.

    .. todo::
        * This should be a read-only class
        * We presently insist that all subroutine groups use the same variable
          index base.  Reasonable?
        * internal_call_graph is used here and in TaskFunction.  It should
          likely become a class that checks the quality of the graph.
        * Milhoja should have an internal parser that can figure out the
          R/W/RW status of each variable of the grid data arrays in each
          subroutine in the internal call graph.
        * Very low-priority optimization is determine if a scratch variable
          for one subroutine can double as scratch variable for another
          subroutine so that we can limit amount of scratch needed by TF.
    """
    __LOG_TAG = "Milhoja TF Assembler"

    @staticmethod
    def from_milhoja_json(name, internal_call_graph,
                          group_jsons_all, grid_json, logger):
        """
        Construct a TaskFunctionAssembler object using Milhoja-JSON format
        subroutine group specification files.

        :param name: Name of the task function
        :param internal_call_graph: Refer to documentation in constructor for
            same argument
        :param group_jsons_all: Filenames with paths of all Milhoja-JSON
            format subroutine group specification files that together contain
            the specifications for the subroutines in the internal call graph
        :param grid_json: Filename with path of Milhoja-JSON format grid
            specification file
        :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
        """
        # ----- ERROR CHECK ARGUMENTS
        # name and internal_call_graph checked by constructor
        if not isinstance(group_jsons_all, list):
            msg = f"group_jsons_all not list ({group_jsons_all})"
            raise TypeError(msg)
        elif not group_jsons_all:
            raise ValueError("No group specifications provided")

        if (not isinstance(grid_json, str)) and \
                (not isinstance(grid_json, Path)):
            raise TypeError("grid_json not string or Path ({grid_json})")
        elif not Path(grid_json).is_file():
            msg = f"{grid_json} does not exist or is not file"
            raise ValueError(msg)

        if not isinstance(logger, AbcLogger):
            raise TypeError("Unknown logger type")

        # ----- LOAD & CONVERT SPECS INTO INTERNAL MILHOJA REPRESENTATION
        group_specs_all = []
        for group_json in group_jsons_all:
            group_spec = SubroutineGroup.from_milhoja_json(group_json, logger)
            group_specs_all.append(group_spec)

        # This is simple enough that I am not going to put a format/version on
        # it for now.
        with open(grid_json, "r") as fptr:
            grid_spec = json.load(fptr)

        return TaskFunctionAssembler(name, internal_call_graph,
                                     group_specs_all, grid_spec, logger)

    def __init__(self, name, internal_call_graph,
                 group_specs_all, grid_spec, logger):
        """
        It is intended that users instantiate assemblers using the from_*
        classmethods.

        :param name: Name of the task function
        :param internal_call_graph: WRITE THIS
        :param group_specs_all:  Set of ``SubroutineGroup`` objects that
            together contain the specifications for the subroutines in the
            internal call graph.  This is stored immediately as a copy so that
            calling code can continue using the actual arguments passed in as
            needed.
        :param grid_spec:  Milhoja-internal grid specification ``dict``.  This
            is stored immediately as a copy so that calling code can continue
            using the actual argument passed in as needed.
        :param logger: Logger derived from :py:class:`milhoja.AbcLogger`
        """
        super().__init__()

        # ----- ERROR CHECK & SETUP LOGGER FOR IMMEDIATE USE
        if not isinstance(logger, AbcLogger):
            raise TypeError("Unknown logger type")
        self.__logger = logger

        # ----- ERROR CHECK OTHER ARGUMENTS
        if not isinstance(name, str):
            raise TypeError(f"name ({name}) is not string")
        self.__tf_name = name

        msg = "Building assembler for task function {}"
        self.__log_debug(msg.format(self.__tf_name))

        self.__call_graph = internal_call_graph

        if not isinstance(group_specs_all, list):
            raise TypeError("group_specs_all not list")
        for group in group_specs_all:
            if not isinstance(group, SubroutineGroup):
                msg = "Items in group_specs_all not all SubroutineGroup"
                raise TypeError(msg)
        self.__group_specs = group_specs_all.copy()

        self.__grid = copy.deepcopy(grid_spec)
        check_grid_specification(self.__grid, self.__logger)

        # Check across all subroutine groups
        #
        # We don't presently handle different index bases
        index_base = self.__group_specs[0].variable_index_base
        for spec in self.__group_specs[1:]:
            if group.variable_index_base != index_base:
                msg = "All subroutine groups must use same variable index base"
                raise NotImplementedError(msg)

        for i, j in it.combinations(range(len(self.__group_specs)), 2):
            group_i = self.__group_specs[i]
            group_j = self.__group_specs[j]
            name_i = group_i.name
            name_j = group_j.name

            # If we allow two groups with the same name and each of these
            # contained group-level external or scratch variables with the same
            # name, then we would have a variable name clash.
            if name_i == name_j:
                msg = "More than one subroutine group with name {} for TF {}"
                raise LogicError(msg.format(name_i, self.__tf_name))

            # Insist that subroutines in each group have names different from
            # those in all other groups so that finding the group that contains
            # a particular subroutine is easy and correct.
            common = set(group_i.subroutines).intersection(group_j.subroutines)
            if common != set():
                msg = "Groups {} & {} specify subroutines with same names ({})"
                raise LogicError(msg.format(name_i, name_j, common))

        # ----- EAGER DETERMINATION OF TF SPECIFICATION
        self.__dummies, self.__dummy_specs, self.__dummy_to_actuals = \
            self.__determine_unique_dummies()

        self.__log_debug("")
        self.__log_debug(f"Task Function {name} Argument Information")
        self.__log_debug("-" * 80)
        for dummy in self.__dummies:
            arg_spec = self.__dummy_specs[dummy]
            arg_type = arg_spec["source"]
            self.__log_debug(f"{dummy} / type {arg_type}")
            for key, value in arg_spec.items():
                if key != "source":
                    self.__log_debug(f"\t{key:<25}{value}")
            self.__log_debug("\tPassed as actual argument to")
            for tmp in self.__dummy_to_actuals[dummy]:
                self.__log_debug(f"\t\t{tmp[1]} in {tmp[0]}")

    def __log(self, msg):
        """
        Log given message at default level
        """
        self.__logger.log(TaskFunctionAssembler.__LOG_TAG, msg,
                          LOG_LEVEL_BASIC)

    def __log_debug(self, msg):
        """
        Log given message at lowest debug log level
        """
        self.__logger.log(TaskFunctionAssembler.__LOG_TAG, msg,
                          LOG_LEVEL_BASIC_DEBUG)

    def __determine_unique_dummies(self):
        """
        This is the workhorse of the assembler that identifies the minimal set
        of dummy arguments for the task function, writes the specifications for
        each dummy, and determines the mapping of task function dummy argument
        onto actual argument for each subroutine in the internal subroutine
        graph.

        .. todo::
            * What about lbound arguments!?

        :return: (tf_dummy_list, tf_dummy_spec, dummy_to_actuals) where

            * **tf_dummy_list :** Ordered dummy argument list for the task
              function
            * **tf_dummy_spec :** Specification of each of the task function's
              dummy arguments
            * **dummy_to_actuals :** Mapping of task function dummy arguments
              onto the arguments of the internal subroutines
        """
        tf_dummy_list = []
        tf_dummy_spec = {}
        dummy_to_actuals = {}

        # While the actual argument ordering likely does not matter.  Make sure
        # that our ordering is fixed so that tests will work on all platforms
        # and all versions of python.
        for arg_specs, arg_mappings in [
            self.__get_external(), self.__get_tile_metadata(),
            self.__get_grid_data(), self.__get_scratch()
        ]:
            assert set(arg_specs).intersection(tf_dummy_spec) == set()
            tf_dummy_spec.update(arg_specs)
            assert set(arg_mappings).intersection(dummy_to_actuals) == set()
            dummy_to_actuals.update(arg_mappings)
            assert set(arg_specs).intersection(tf_dummy_list) == set()
            tf_dummy_list += sorted(arg_specs.keys())

        # The Milhoja thread index is immediately available since the calling
        # thread passes it when it calls the TF.  Therefore, it should not
        # appear in the TF argument list.  However, we need to correctly
        # specify the argument of the internal subroutines that need it
        # as the actual argument.
        arg_mappings = self.__get_milhoja_thread_index()
        assert set(arg_mappings).intersection(dummy_to_actuals) == set()
        dummy_to_actuals.update(arg_mappings)

        return tf_dummy_list, tf_dummy_spec, dummy_to_actuals

    @property
    def task_function_name(self):
        return self.__tf_name

    @property
    def __internal_subroutine_graph(self):
        """
        :return: Generator for iterating in correct order over the nodes in the
            internal subroutine graph of the task function.  Each node contains
            one or more subroutines.  If more than one, it is understood that
            the subroutines in that node can be run concurrently.  Assume that
            this accesses each subroutine within a single node in arbitrary
            order.
        """
        for node in self.__call_graph:
            if isinstance(node, str):
                yield [node]
            else:
                yield node

    def __find_subroutine_group(self, subroutine):
        """
        """
        for group in self.__group_specs:
            if subroutine in group:
                return group

        msg = f"Subroutine {subroutine} not specified in any group"
        raise LogicError(msg)

    @property
    def dummy_arguments(self):
        """
        :return: List of dummy arguments of assembled task function given in
            final ordering
        """
        return self.__dummies

    def argument_specification(self, argument):
        """
        :return: Deep copy of specification of given task function dummy
            argument
        """
        if argument not in self.__dummy_specs:
            name = self.task_function_name
            msg = f"{argument} is not dummy argument for TF {name}"
            raise ValueError(msg)

        return copy.deepcopy(self.__dummy_specs[argument])

    @property
    def variable_index_base(self):
        """
        :return: All variable indices provided in the variable-in/-out fields
            of the task function specification are part of an index set whose
            smallest index is this value.  Valid values are either 0 or 1.
        """
        return self.__group_specs[0].variable_index_base

    def __get_milhoja_thread_index(self):
        """
        Runs through the each subroutine in the internal call graph to
        determine if any of the internal subroutines require the unique index
        of the Milhoja thread calling the TF as an actual argument.  If so,
        determines onto which actual arguments it should be mapped.

        :return: Dummy to actual arguments mapping
        """
        dummy_to_actuals = {}
        for node in self.__internal_subroutine_graph:
            for subroutine in node:
                group = self.__find_subroutine_group(subroutine)
                for idx, arg in enumerate(group.argument_list(subroutine)):
                    arg_spec = group.argument_specification(subroutine, arg)
                    source = arg_spec["source"]
                    if source.lower() == THREAD_INDEX_ARGUMENT:
                        # Use same variable name used by other Milhoja tools
                        tf_dummy = THREAD_INDEX_VAR_NAME
                        if tf_dummy not in dummy_to_actuals:
                            dummy_to_actuals[tf_dummy] = []
                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        return dummy_to_actuals

    def __get_external(self):
        """
        Runs through the each subroutine in the internal call graph, determines
        the minimum set of external arguments that need to be included in the
        task function dummy arguments, assembles the specifications for these
        dummies, and determines how to map each task function dummy argument
        onto the actual argument list of each internal subroutine.

        Note that external dummy variables are named::

                       external_<operation name>_<variable name>

        to avoid collisions if the task function needs two variables with the
        same name but from two different operations.  Note that the prefix is
        needed to avoid collisions with scratch variables as well.

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of external dummy arguments
              for the task function and
            * dummy_to_actuals is the mapping
        """
        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.__internal_subroutine_graph:
            for subroutine in node:
                group = self.__find_subroutine_group(subroutine)
                for idx, arg in enumerate(group.argument_list(subroutine)):
                    arg_spec = group.argument_specification(subroutine, arg)
                    source = arg_spec["source"]
                    if source == EXTERNAL_ARGUMENT:
                        global_name = arg_spec["name"].strip()
                        assert global_name.startswith("_")
                        tf_dummy = f"external_{group.name}{global_name}"
                        if tf_dummy not in tf_dummy_spec:
                            # This is deep copy, so we can alter without
                            # altering original specification
                            external_spec = \
                                group.external_specification(global_name)
                            assert "source" not in external_spec
                            external_spec["source"] = EXTERNAL_ARGUMENT
                            tf_dummy_spec[tf_dummy] = external_spec
                            assert tf_dummy not in dummy_to_actuals
                            dummy_to_actuals[tf_dummy] = []

                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        return tf_dummy_spec, dummy_to_actuals

    def __get_tile_metadata(self):
        """
        Runs through the each subroutine in the internal call graph, determines
        the minimum set of tile metadata arguments that need to be included in
        the task function dummy arguments, assembles the specifications for
        these dummies, and determines how to map each task function dummy
        argument onto the actual argument list of each internal subroutine.

        If two subroutines need the same geometry data (i.e., coordinates,
        areas, or volumes) but over different overlapping regions, then we
        include the data as a single variable that contains data in a region
        large enough that it contains all data needed by both.  This implies
        that all subroutines must be written to accept geometry data in arrays
        potentially larger than what they need.

        .. todo::
            * Should this class include a different geometry variable for each
              different region?  It seems like it should.  We could then add
              on an additional optimization layer that can do this merging.

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of tile metadata dummy
              arguments for the task function and
            * dummy_to_actuals is the mapping
        """
        REQUIRE_MERGING = [TILE_COORDINATES_ARGUMENT,
                           TILE_FACE_AREAS_ARGUMENT,
                           TILE_CELL_VOLUMES_ARGUMENT]
        AXIS_LUT = {"i": "x", "j": "y", "k": "z"}

        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.__internal_subroutine_graph:
            for subroutine in node:
                group = self.__find_subroutine_group(subroutine)
                for idx, arg in enumerate(group.argument_list(subroutine)):
                    arg_spec = group.argument_specification(subroutine, arg)
                    source = arg_spec["source"]
                    if source in TILE_ARGUMENTS_ALL:
                        if len(arg_spec) == 1:
                            tf_dummy = source
                        else:
                            if source == TILE_COORDINATES_ARGUMENT:
                                axis = AXIS_LUT[arg_spec["axis"].lower()]
                                edge = arg_spec["edge"].lower()
                                tf_dummy = f"tile_{axis}Coords_{edge}"
                            elif source == TILE_FACE_AREAS_ARGUMENT:
                                axis = AXIS_LUT[arg_spec["axis"].lower()]
                                tf_dummy = f"tile_{axis}FaceAreas"
                            elif source == TILE_CELL_VOLUMES_ARGUMENT:
                                tf_dummy = TILE_CELL_VOLUMES_ARGUMENT
                            else:
                                msg = "Unhandled tile argument {} of type {}"
                                raise LogicError(msg.format(arg, source))

                        if tf_dummy not in tf_dummy_spec:
                            tf_dummy_spec[tf_dummy] = arg_spec
                            assert tf_dummy not in dummy_to_actuals
                            dummy_to_actuals[tf_dummy] = []
                        elif source in REQUIRE_MERGING:
                            if arg_spec["lo"] == TILE_LBOUND_ARGUMENT:
                                tf_dummy_spec[tf_dummy]["lo"] = \
                                    TILE_LBOUND_ARGUMENT
                            if arg_spec["hi"] == TILE_UBOUND_ARGUMENT:
                                tf_dummy_spec[tf_dummy]["hi"] = \
                                    TILE_UBOUND_ARGUMENT

                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        return tf_dummy_spec, dummy_to_actuals

    def __grid_data_name(self, index_space, structure_index):
        """
        :param index_space: Index space of grid data structure.  Valid values
            are CENTER and FLUX[XYZ].
        :param structure_index: 1-based index into array of grid data structures
            of given index space.

        :return: Unique name of dummy parameter to be associated consistently
            with given grid data structure
        """
        SPACES_MAPPING = {
            "center": "CC_{0}",
            "fluxx": "FLX_{0}",
            "fluxy": "FLY_{0}",
            "fluxz": "FLZ_{0}"
        }

        space = index_space.lower()
        if space not in SPACES_MAPPING:
            raise ValueError(f"Invalid grid index space ({index_space})")
        if structure_index != 1:
            msg = "Only one {} data structure presently allowed"
            raise NotImplementedError(msg.format(index_space))

        return SPACES_MAPPING[space].format(structure_index)

    def determine_access_patterns(self):
        """
        For each grid data structure used by the task function, identify which
        of its variables are used and the access pattern for each of these
        throughout the execution of the internal call graph.

        This functionality is placed in a public member function for testing
        purposes.

        .. todo::
            * Add error checking to make sure that all subroutines within a
              single node use variables in a mutually consistent, correct way?

        :return: Doubly-nested dict.  First key is dummy argument; second,
            variable name.  Each value is the ordered list of access patterns.
            A result of ["R", "RW", "W", "R"] indicates that as the subroutines
            are executed in correct order, the variable was first read from,
            then written two a few times, and finally read from.
        """
        accesses = {}
        for node in self.__internal_subroutine_graph:
            for subroutine in node:
                group = self.__find_subroutine_group(subroutine)
                for idx, arg in enumerate(group.argument_list(subroutine)):
                    arg_spec = group.argument_specification(subroutine, arg)
                    source = arg_spec["source"]
                    if source == GRID_DATA_ARGUMENT:
                        tf_dummy = self.__grid_data_name(
                                        *arg_spec["structure_index"]
                                   )
                        if tf_dummy not in accesses:
                            accesses[tf_dummy] = {}

                        for access in ["R", "RW", "W"]:
                            if access in arg_spec:
                                for idx in arg_spec[access]:
                                    if idx not in accesses[tf_dummy]:
                                        accesses[tf_dummy][idx] = []
                                    accesses[tf_dummy][idx].append(access)

        return accesses

    def determine_variable_masks(self, variable_accesses):
        """
        This functionality is placed in a public member function for testing
        purposes.

        .. note::
            * This assumes that a variable that is RW for the first subroutine
              that uses it is read before it is written to and, therefore, is
              marked as an 'in' variable.  This seems reasonable since
              otherwise the variable should likely be specified as W.
            * This assumes that a variable that is written to at any time should
              be marked as an 'out' variable regardless of how it is used
              afterward.  While this might not always be true, this tool does
              not presently have a means to determine this.

        :param variable_accesses: Result from determine_access_patterns()

        :return: Doubly-nested dict.  First key is dummy argument; second,
            variable type.  Each value is (min, max) mask where min is the
            smallest variable needed of the type; max, the largest.
        """
        VARIABLES_IN = "variables_in"
        VARIABLES_OUT = "variables_out"

        masks = {}
        for dummy, variables_all in variable_accesses.items():
            vars_in = set()
            vars_out = set()
            for variable, accesses in variables_all.items():
                assert all([e in ["R", "RW", "W"] for e in accesses])
                if accesses[0] in ["R", "RW"]:
                    vars_in.add(variable)
                else:
                    vars_out.add(variable)

                if any([e in ["RW", "W"] for e in accesses]):
                    vars_out.add(variable)

            assert dummy not in masks
            masks[dummy] = {}
            if vars_in:
                masks[dummy][VARIABLES_IN] = [min(vars_in), max(vars_in)]
            if vars_out:
                masks[dummy][VARIABLES_OUT] = [min(vars_out), max(vars_out)]

        return masks

    def __get_grid_data(self):
        """
        Runs through the each subroutine in the internal call graph, determines
        the minimum set of grid-managed data structure arguments that need to
        be included in the task function dummy arguments, assembles the
        specifications for these dummies, and determines how to map each task
        function dummy argument onto the actual argument list of each internal
        subroutine.  This also determines the variable-in/-out masks for each
        dummy argument.

        .. note::
            * This presently supports only one grid data structure per index
              space
            * The variable masks are constructed without needing to know if
              variable indexing is 0 or 1 based

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of grid data dummy arguments
              for the task function and
            * dummy_to_actuals is the mapping
        """
        VARIABLES_IN = "variables_in"
        VARIABLES_OUT = "variables_out"

        # ----- DETERMINE DUMMY SPECS WITHOUT VARIABLE MASKS
        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.__internal_subroutine_graph:
            for subroutine in node:
                group = self.__find_subroutine_group(subroutine)
                for idx, arg in enumerate(group.argument_list(subroutine)):
                    arg_spec = group.argument_specification(subroutine, arg)
                    source = arg_spec["source"]
                    if source == GRID_DATA_ARGUMENT:
                        tf_dummy = self.__grid_data_name(
                                        *arg_spec["structure_index"]
                                   )
                        if tf_dummy not in tf_dummy_spec:
                            tf_dummy_spec[tf_dummy] = arg_spec
                            assert tf_dummy not in dummy_to_actuals
                            dummy_to_actuals[tf_dummy] = []
                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        # ----- REPLACE R/RW/W INFO WITH VARIABLE MASKS
        # tf_dummy_spec values are set to deep copies, so we can change below
        # without altering original specifications
        variable_accesses = self.determine_access_patterns()
        variable_masks = self.determine_variable_masks(variable_accesses)
        for dummy in tf_dummy_spec:
            for each in ["R", "RW", "W"]:
                if each in tf_dummy_spec[dummy]:
                    del tf_dummy_spec[dummy][each]

            mask = variable_masks[dummy]
            assert VARIABLES_IN not in tf_dummy_spec[dummy]
            assert VARIABLES_OUT not in tf_dummy_spec[dummy]
            if VARIABLES_IN in mask:
                tf_dummy_spec[dummy][VARIABLES_IN] = mask[VARIABLES_IN]
            if VARIABLES_OUT in mask:
                tf_dummy_spec[dummy][VARIABLES_OUT] = mask[VARIABLES_OUT]

        return tf_dummy_spec, dummy_to_actuals

    def __get_scratch(self):
        """
        Runs through each subroutine in the internal call graph, determines the
        minimum set of scratch arguments that need to be included in the task
        function dummy arguments, assembles the specifications for these
        dummies, and determines how to map each task function dummy argument
        onto the actual argument list of each internal subroutine.

        Note that scratch dummy variables are named::

                         scratch_<operation name>_<variable name>

        to avoid collisions if the task function needs two variables with the
        same name but from two different operations.  Note that the prefix is
        needed to avoid collisions with external variables as well.

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of scratch dummy arguments for
              the task function and
            * dummy_to_actuals is the mapping
        """
        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.__internal_subroutine_graph:
            for subroutine in node:
                group = self.__find_subroutine_group(subroutine)
                for idx, arg in enumerate(group.argument_list(subroutine)):
                    arg_spec = group.argument_specification(subroutine, arg)
                    source = arg_spec["source"]
                    if source == SCRATCH_ARGUMENT:
                        global_name = arg_spec["name"].strip()
                        assert global_name.startswith("_")
                        tf_dummy = f"scratch_{group.name}{global_name}"
                        if tf_dummy not in tf_dummy_spec:
                            # scratch_spec is a deep copy, so we can change
                            # without altering original specification
                            scratch_spec = \
                                group.scratch_specification(global_name)
                            assert "source" not in scratch_spec
                            scratch_spec["source"] = SCRATCH_ARGUMENT
                            tf_dummy_spec[tf_dummy] = scratch_spec
                            assert tf_dummy not in dummy_to_actuals
                            dummy_to_actuals[tf_dummy] = []

                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        return tf_dummy_spec, dummy_to_actuals

    @property
    def tile_metadata_arguments(self):
        """
        :return: Set of task function's dummy arguments that are classified as
            tile metatdata
        """
        dummies = set()
        for arg, spec in self.__dummy_specs.items():
            if spec["source"] in TILE_ARGUMENTS_ALL:
                dummies.add(arg)
        return dummies

    @property
    def external_arguments(self):
        """
        :return: Set of task function's dummy arguments that are classified as
            external arguments
        """
        dummies = set()
        for arg, spec in self.__dummy_specs.items():
            if spec["source"] == EXTERNAL_ARGUMENT:
                dummies.add(arg)
        return dummies

    @property
    def grid_data_structures(self):
        """
        .. todo::
            * Should this include variable-in/-out information as well?

        :return: Set of grid data structures whose data are read from or set by
            the task function
        """
        dummies = {}
        for arg, spec in self.__dummy_specs.items():
            if spec["source"] == GRID_DATA_ARGUMENT:
                space, struct_index = spec["structure_index"]
                if space not in dummies:
                    dummies[space] = {struct_index}
                else:
                    msg = "Only one structure per index space"
                    raise NotImplementedError(msg)
        return dummies

    @property
    def scratch_arguments(self):
        """
        :return: Set of task function's dummy arguments that are classified as
            scratch arguments
        """
        dummies = set()
        for arg, spec in self.__dummy_specs.items():
            if spec["source"] == SCRATCH_ARGUMENT:
                dummies.add(arg)
        return dummies

    def __sanity_check_tf_spec(self, spec):
        """
        If this does not raise an error, then the specification is acceptable.

        While this *does* requires that all specifications be provided, it does
        *not* check the values of specifications that are not used.  For
        example, a C++ TF can specify any value for the fortran_source value.

        .. todo::
            * This should also check types
            * Eventually we might have DataPackets sent to CPUs, in which case
              we would want the specification to give the byte alignment as
              well.
            * Allow applications to not specify unnecessary values?  For
              example, do C++ applications always have to specify (with
              whatever value) the Fortran-specific info?
        """
        # ----- ROOT
        expected = {"task_function", "data_item"}
        actual = set(spec)
        if actual != expected:
            msg = f"Invalid root specification keys ({actual})"
            raise ValueError(msg)
        tf_spec = spec["task_function"]
        data_item = spec["data_item"]

        # ----- TASK FUNCTION
        expected = {"language", "processor",
                    "cpp_header", "cpp_source",
                    "c2f_source", "fortran_source"}
        actual = set(tf_spec)
        if actual != expected:
            msg = f"Invalid TF specification keys ({actual})"
            raise ValueError(msg)
        language = tf_spec["language"]
        processor = tf_spec["processor"]

        if language.lower() not in ["c++", "fortran"]:
            raise ValueError(f"Unsupported TF language ({language})")
        if processor.lower() not in ["cpu", "gpu"]:
            raise ValueError(f"Unsupported target processor ({processor})")
        for each in ["cpp_header", "cpp_source"]:
            if tf_spec[each] == "":
                raise ValueError(f"Empty {each} filename")
        if language.lower() == "fortran":
            for each in ["c2f_source", "fortran_source"]:
                if tf_spec[each] == "":
                    raise ValueError(f"Empty {each} filename")

        # ----- DATA ITEM
        expected = {"type", "byte_alignment", "header", "source"}
        actual = set(data_item)
        if actual != expected:
            msg = f"Invalid data item specification keys ({actual})"
            raise ValueError(msg)

        item_type = data_item["type"]
        if item_type.lower() not in ["tilewrapper", "datapacket"]:
            msg = f"Unsupported data item type ({item_type})"
            raise ValueError(msg)

        for each in ["header", "source"]:
            if data_item[each] == "":
                raise ValueError(f"Empty {each} filename")

        if item_type.lower() == "datapacket":
            byte_align = data_item["byte_alignment"]
            if byte_align <= 0:
                raise ValueError("Non-positive byte alignment ({byte_align})")

    def to_milhoja_json(self, filename, tf_spec_filename, overwrite):
        """
        Write the assembled task function to the given file using the current
        version of the Milhoja-JSON task function specification format.

        .. todo::
            * Milhoja should have an internal parser that gets the argument
              list for each subroutine in the internal call graph.  Then the
              given subroutine JSON files don't need to specify that.
            * Log arguments as debug information

        :param filename: Name and full path of file to write to
        :param tf_spec_filename: Name and full path of file that contains
            concrete task function specification
        :param overwrite: Raise exception if the file exists and this is True
        """
        msg = "Writing {} spec to Milhoja-JSON file {}"
        self.__log(msg.format(self.task_function_name, filename))

        if not Path(tf_spec_filename).is_file():
            msg = f"{tf_spec_filename} does not exist or is not a file"
            raise ValueError(msg)
        with open(tf_spec_filename, "r") as fptr:
            tf_spec = json.load(fptr)
        self.__sanity_check_tf_spec(tf_spec)

        spec = {}
        spec["format"] = [MILHOJA_JSON_FORMAT, CURRENT_MILHOJA_JSON_VERSION]

        # ----- INCLUDE GRID SPECIFICATION
        assert "grid" not in spec
        spec["grid"] = self.__grid

        # ----- INCLUDE TASK FUNCTION SPECIFICATION
        outer = "task_function"
        assert outer not in spec
        spec[outer] = tf_spec[outer]
        spec[outer]["name"] = self.task_function_name
        spec[outer]["argument_list"] = self.dummy_arguments
        spec[outer]["variable_index_base"] = self.variable_index_base

        key = "argument_specifications"
        assert key not in spec[outer]
        spec[outer][key] = {}
        for dummy in self.dummy_arguments:
            assert dummy not in spec[outer][key]
            spec[outer][key][dummy] = self.argument_specification(dummy)

        key = "subroutine_call_graph"
        assert key not in spec[outer]
        spec[outer][key] = self.__call_graph

        # ----- INCLUDE DATA ITEM SPECIFICATION
        outer = "data_item"
        assert outer not in spec
        spec[outer] = tf_spec[outer]

        # ----- INCLUDE SUBROUTINES SPECIFICATION
        outer = "subroutines"
        assert outer not in spec
        spec[outer] = {}
        for node in self.__internal_subroutine_graph:
            for subroutine in node:
                group = self.__find_subroutine_group(subroutine)
                sub_spec = group.subroutine_specification(subroutine)
                dummies = group.argument_list(subroutine)

                mapping = {}
                for arg in dummies:
                    for key, value in self.__dummy_to_actuals.items():
                        for item in value:
                            item = (item[0], item[1])
                            if item == (subroutine, arg):
                                assert arg not in mapping
                                mapping[arg] = key

                assert subroutine not in spec[outer]
                spec[outer][subroutine] = {
                    "interface_file": sub_spec["interface_file"],
                    "argument_list": dummies,
                    "argument_mapping": mapping
                }

        if (not overwrite) and Path(filename).exists():
            raise RuntimeError(f"{filename} already exists")

        with open(filename, "w") as fptr:
            json.dump(
                spec, fptr, ensure_ascii=True, allow_nan=False, indent="\t"
            )
