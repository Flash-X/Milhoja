import json

from pathlib import Path

from . import MILHOJA_JSON_FORMAT
from . import CURRENT_MILHOJA_JSON_VERSION
from . import LOG_LEVEL_BASIC
from . import LOG_LEVEL_BASIC_DEBUG
# from . import AbcLogger
from . import TaskFunction
from . import check_operation_specification


class TaskFunctionAssembler(object):
    """
    .. todo::
        * Write documentation here and for all methods in class.
        * internal_call_graph is used here and in TaskFunction.  It should
          likely become a class.
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
                          operation_json, logger):
        """
        .. todo::
            * Load JSON files carefully and accounting for different versions
              as is done in TaskFunction.from_milhoja_json.
            * If the internal call graph includes subroutines from different
              operations, then we likely need multiple op specs.
            * Log arguments as debug information

        :param name: Name of the task function
        :param internal_call_graph: Refer to documentation in constructor for
            same argument
        :param operation_json: WRITE THIS
        :param logger: Logger derived from milhoja.AbcLogger
        """
        # if not isinstance(logger, AbcLogger):
        #     raise TypeError("Unknown logger type")
        msg = f"Loading {name} Milhoja-JSON file {operation_json}"
        logger.log(TaskFunctionAssembler.__LOG_TAG, msg, LOG_LEVEL_BASIC)

        if not Path(operation_json).is_file():
            msg = f"{operation_json} does not exist or is not a file"
            raise ValueError(msg)
        with open(operation_json, "r") as fptr:
            operation_spec = json.load(fptr)

        return TaskFunctionAssembler(name, internal_call_graph,
                                     operation_spec, logger)

    def __init__(self, name, internal_call_graph,
                 operation_spec, logger):
        """
        It is intended that users instantiate assemblers using the from_*
        classmethods.

        .. todo::
            * Error check graph and each specification here with informative
              error messages
            * Log arguments as debug information

        :param name: Name of the task function
        :param internal_call_graph: WRITE THIS
        :param operation_spec: WRITE THIS
        :param logger: Logger derived from milhoja.AbcLogger
        """
        super().__init__()

        # if not isinstance(logger, AbcLogger):
        #     raise TypeError("Unknown logger type")
        self.__logger = logger

        msg = f"Building assembler for task function {name}"
        self.__logger.log(TaskFunctionAssembler.__LOG_TAG, msg,
                          LOG_LEVEL_BASIC_DEBUG)

        self.__tf_name = name
        self.__call_graph = internal_call_graph
        self.__op_spec = operation_spec

        # ----- ERROR CHECK ALL INPUTS
        check_operation_specification(self.__op_spec)

        # ----- EAGER DETERMINATION OF TF SPECIFICATION
        self.__dummies, self.__dummy_specs, self.__dummy_to_actuals = \
            self.__determine_unique_dummies(self.__op_spec)

    def __sanity_check_subroutine(self, name, spec):
        """
        .. todo::
            * Finish checks
            * Move this out as own routine in package
        """
        expected = {
            "interface_file", "argument_list", "argument_specifications"
        }
        actual = set(spec)
        if actual != expected:
            msg = f"Invalid set of {name} specification keys ({actual})"
            raise ValueError(msg)
        interface = spec["interface_file"]
        arg_list = spec["argument_list"]
        arg_specs_all = spec["argument_specifications"]

        if interface == "":
            raise ValueError(f"Empty {name} interface filename")
        if len(arg_list) != len(arg_specs_all):
            msg = f"Incompatible argument list & specification for {name}"
            raise ValueError(msg)
        for arg in arg_list:
            arg_spec = arg_specs_all[arg]
            if "source" not in arg_spec:
                msg = f"{arg} in {name} missing source field"
                raise ValueError(msg)
            source = arg_spec["source"]
            is_thread_idx = (source == TaskFunction.THREAD_INDEX_ARGUMENT)
            is_external = (source == TaskFunction.EXTERNAL_ARGUMENT)
            is_tile_metadata = (source in TaskFunction.TILE_METADATA_ALL)
            is_grid = (source == TaskFunction.GRID_DATA_ARGUMENT)
            is_scratch = (source == TaskFunction.SCRATCH_ARGUMENT)
            if (not is_thread_idx) and \
                    (not is_external) and \
                    (not is_tile_metadata) and \
                    (not is_grid) and \
                    (not is_scratch):
                msg = f"{arg} in {name} has unknown source {source}"
                raise ValueError(msg)

    def __sanity_check_grid_spec(self, op_spec):
        """
        If this does not raise an error, then the specification is acceptable.

        .. todo::
            * This should also check types
            * Move this out as own routine in package
        """
        if "grid" not in op_spec:
            msg = "Operation specification missing grid specification"
            raise ValueError(msg)

        grid_spec = op_spec["grid"]

        expected = {"dimension", "nxb", "nyb", "nzb", "nguardcells"}
        actual = set(grid_spec)
        if actual != expected:
            msg = f"Invalid set of grid specification keys ({actual})"
            raise ValueError(msg)

        dimension = grid_spec["dimension"]
        if dimension not in [1, 2, 3]:
            msg = f"Invalid grid dimension ({dimension})"
            raise ValueError(msg)

        nxb = grid_spec["nxb"]
        if nxb <= 0:
            raise ValueError(f"Non-positive NXB ({nxb})")

        nyb = grid_spec["nyb"]
        if nyb <= 0:
            raise ValueError(f"Non-positive NYB ({nyb})")
        elif (dimension == 1) and (nyb != 1):
            raise ValueError("nyb > 1 for 1D problem")

        nzb = grid_spec["nzb"]
        if nzb <= 0:
            raise ValueError(f"Non-positive NZB ({nzb})")
        elif (dimension < 3) and (nzb != 1):
            raise ValueError(f"nzb > 1 for {dimension}D problem")

        n_gc = grid_spec["nguardcells"]
        if n_gc < 0:
            raise ValueError(f"Negative N guardcells ({n_gc})")

    def __determine_unique_dummies(self, op_spec):
        """
        This is the workhorse of the assembler that identifies the minimal set
        of dummy arguments for the task function, writes the specifications for
        each dummy, and determines the mapping of task function dummy argument
        onto actual argument for each subroutine in the internal subroutine
        graph.

        .. todo::
            * What about lbound arguments!?
        """
        tf_dummy_list = []
        tf_dummy_spec = {}
        dummy_to_actuals = {}

        # While the actual argument ordering likely does not matter.  Make sure
        # that our ordering is fixed so that tests will work on all platforms
        # and all versions of python.
        for arg_specs, arg_mappings in [
            self.__get_external(op_spec), self.__get_tile_metadata(),
            self.__get_grid_data(), self.__get_scratch(op_spec)
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
    def internal_subroutine_graph(self):
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

    def subroutine_specification(self, subroutine):
        """
        :return: Specification of given subroutine as a dictionary
        """
        if subroutine not in self.__op_spec["operation"]:
            name = self.task_function_name
            msg = "{} not specified in any operation for TF {}"
            raise ValueError(msg.format(subroutine, name))

        return self.__op_spec["operation"][subroutine]

    @property
    def dummy_arguments(self):
        """
        :return: List of dummy arguments of assembled task function given in
            final ordering
        """
        return self.__dummies

    def argument_specification(self, argument):
        """
        :return: Specification of given task function dummy argument
        """
        if argument not in self.__dummy_specs:
            name = self.task_function_name
            msg = f"{argument} is not dummy argument for TF {name}"
            raise ValueError(msg)

        return self.__dummy_specs[argument]

    @property
    def variable_index_base(self):
        """
        .. todo::
            * Once we have multiple operation specs, do not assume that each
              will use the same base.  How to manage this difficulty?

        :return: All variable indices provided in the variable-in/-out fields
            of the task function specification are part of an index set whose
            smallest index is this value.  Valid values are either 0 or 1.
        """
        return self.__op_spec["operation"]["variable_index_base"]

    def __get_milhoja_thread_index(self):
        """
        Runs through the each subroutine in the internal call graph to
        determine if any of the internal subroutines require the unique index
        of the Milhoja thread calling the TF as an actual argument.  If so,
        determines onto which actual arguments it should be mapped.

        :return: Dummy to actual arguments mapping
        """
        dummy_to_actuals = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source.lower() == TaskFunction.THREAD_INDEX_ARGUMENT:
                        # Use same variable name used by other Milhoja tools
                        tf_dummy = TaskFunction.THREAD_INDEX_VAR_NAME
                        if tf_dummy not in dummy_to_actuals:
                            dummy_to_actuals[tf_dummy] = []
                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        return dummy_to_actuals

    def __get_external(self, op_spec):
        """
        Runs through the each subroutine in the internal call graph, determines
        the minimum set of external arguments that need to be included in the
        task function dummy arguments, assembles the specifications for these
        dummies, and determines how to map each task function dummy argument
        onto the actual argument list of each internal subroutine.

        Note that external dummy variables are named::

                         <operation name>_<variable name>

        to avoid collisions if the task function needs two variables with the
        same name but from two different operations.

        .. todo::
            * This is presently restricted to a single operation.

        :param bridge: Data structure that specifies the scratch and external
            arguments at the level of the single operation to which all
            internal subroutines belong.
        :param operation_name: Name of the operation

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of external dummy arguments
              for the task function and
            * dummy_to_actuals is the mapping
        """
        op_name = op_spec["operation"]["name"]

        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    # Get copy since we might alter below for internal
                    # purposes only
                    arg_spec = spec["argument_specifications"][arg].copy()
                    source = arg_spec["source"]
                    if source == TaskFunction.EXTERNAL_ARGUMENT:
                        external_spec = op_spec["operation"]["external"]
                        op_external_name = arg_spec["name"].strip()
                        assert op_external_name.startswith("_")
                        tf_dummy = f"{op_name}{op_external_name}"
                        if tf_dummy not in tf_dummy_spec:
                            tmp_spec = external_spec[op_external_name].copy()
                            assert "source" not in tmp_spec
                            tmp_spec["source"] = "external"
                            tf_dummy_spec[tf_dummy] = tmp_spec
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

        .. todo::
            * Certain tile_metadata vars may only need a subset of their
              arguments checked in order to assume that the var has already
              been accounted for.
            * Always use a fixed variable name for metadata that requires
              extra specs.
            * If one subroutine needs cell volumes in interior and another
              needs them on the full block, do we include just the larger one
              or both?

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of tile metadata dummy
              arguments for the task function and
            * dummy_to_actuals is the mapping
        """
        axis_lut = {"i": "x", "j": "y", "k": "z"}

        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source in TaskFunction.TILE_METADATA_ALL:
                        if len(arg_spec) == 1:
                            tf_dummy = source
                        else:
                            if source.lower() == "tile_coordinates":
                                axis = axis_lut[arg_spec["axis"].lower()]
                                edge = arg_spec["edge"].lower()
                                tf_dummy = f"tile_{axis}Coords_{edge}"
                            elif source.lower() == "tile_cellvolumes":
                                tf_dummy = "tile_cellVolumes"
                            else:
                                print(arg, arg_spec)
                                raise NotImplementedError("Test case!")

                        if tf_dummy not in tf_dummy_spec:
                            tf_dummy_spec[tf_dummy] = arg_spec
                            assert tf_dummy not in dummy_to_actuals
                            dummy_to_actuals[tf_dummy] = []
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
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source == TaskFunction.GRID_DATA_ARGUMENT:
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
              marked as an in variable.  This seems reasonable.
            * This assumes that a variable that is written to at any time should
              be marked as an out variable regardless of how it is used
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
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source == TaskFunction.GRID_DATA_ARGUMENT:
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
        variable_accesses = self.determine_access_patterns()
        variable_masks = self.determine_variable_masks(variable_accesses)
        for dummy, variables_all in tf_dummy_spec.items():
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

    def __get_scratch(self, op_spec):
        """
        Runs through each subroutine in the internal call graph, determines the
        minimum set of scratch arguments that need to be included in the task
        function dummy arguments, assembles the specifications for these
        dummies, and determines how to map each task function dummy argument
        onto the actual argument list of each internal subroutine.

        Note that scratch dummy variables are named::

                         <operation name>_<variable name>

        to avoid collisions if the task function needs two variables with the
        same name but from two different operations.

        :param bridge: Data structure that specifies the scratch and external
            arguments at the level of the single operation to which all
            internal subroutines belong.
        :param operation_name: Name of the operation

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of scratch dummy arguments for
              the task function and
            * dummy_to_actuals is the mapping
        """
        op_name = op_spec["operation"]["name"]

        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source == TaskFunction.SCRATCH_ARGUMENT:
                        scratch_spec = op_spec["operation"]["scratch"]
                        op_scratch_name = arg_spec["name"].strip()
                        assert op_scratch_name.startswith("_")
                        tf_dummy = f"{op_name}{op_scratch_name}"
                        if tf_dummy not in tf_dummy_spec:
                            tmp_spec = scratch_spec[op_scratch_name].copy()
                            assert "source" not in tmp_spec
                            tmp_spec["source"] = "scratch"
                            tf_dummy_spec[tf_dummy] = tmp_spec
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
            if spec["source"] in TaskFunction.TILE_METADATA_ALL:
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
            if spec["source"] == TaskFunction.EXTERNAL_ARGUMENT:
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
            if spec["source"] == TaskFunction.GRID_DATA_ARGUMENT:
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
            if spec["source"] == TaskFunction.SCRATCH_ARGUMENT:
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
        msg = msg.format(self.task_function_name, filename)
        self.__logger.log(TaskFunctionAssembler.__LOG_TAG, msg, LOG_LEVEL_BASIC)

        if not Path(tf_spec_filename).is_file():
            msg = f"{tf_spec_filename} does not exist or is not a file"
            raise ValueError(msg)
        with open(tf_spec_filename, "r") as fptr:
            tf_spec = json.load(fptr)
        self.__sanity_check_tf_spec(tf_spec)

        spec = {}
        spec["format"] = [MILHOJA_JSON_FORMAT, CURRENT_MILHOJA_JSON_VERSION]

        # ----- INCLUDE GRID SPECIFICATION
        group = "grid"
        assert group not in spec
        spec[group] = self.__op_spec["grid"]

        # ----- INCLUDE TASK FUNCTION SPECIFICATION
        group = "task_function"
        assert group not in spec
        spec[group] = tf_spec[group]
        spec[group]["name"] = self.task_function_name
        spec[group]["argument_list"] = self.dummy_arguments
        spec[group]["variable_index_base"] = self.variable_index_base

        key = "argument_specifications"
        assert key not in spec[group]
        spec[group][key] = {}
        for dummy in self.dummy_arguments:
            assert dummy not in spec[group][key]
            spec[group][key][dummy] = self.argument_specification(dummy)

        key = "subroutine_call_graph"
        assert key not in spec[group]
        spec[group][key] = self.__call_graph

        # ----- INCLUDE DATA ITEM SPECIFICATION
        group = "data_item"
        assert group not in spec
        spec[group] = tf_spec[group]

        # ----- INCLUDE SUBROUTINES SPECIFICATION
        group = "subroutines"
        assert group not in spec
        spec[group] = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                subroutine_spec = self.subroutine_specification(subroutine)
                dummies = subroutine_spec["argument_list"]

                mapping = {}
                for arg in dummies:
                    for key, value in self.__dummy_to_actuals.items():
                        for item in value:
                            item = (item[0], item[1])
                            if item == (subroutine, arg):
                                assert arg not in mapping
                                mapping[arg] = key

                assert subroutine not in spec[group]
                spec[group][subroutine] = {
                    "interface_file": subroutine_spec["interface_file"],
                    "argument_list": dummies,
                    "argument_mapping": mapping
                }

        if (not overwrite) and Path(filename).exists():
            raise RuntimeError(f"{filename} already exists")

        with open(filename, "w") as fptr:
            json.dump(
                spec, fptr, ensure_ascii=True, allow_nan=False, indent="\t"
            )
