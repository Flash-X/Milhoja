import json

from pathlib import Path

from . import MILHOJA_JSON_FORMAT
from . import CURRENT_MILHOJA_JSON_VERSION
from . import LOG_LEVEL_BASIC
from . import LOG_LEVEL_BASIC_DEBUG
# from . import AbcLogger
from . import TaskFunction


class TaskFunctionAssembler(object):
    """
    .. todo::
        * Write documentation here and for all methods in class.
        * internal_call_graph is used here and in TaskFunction.  It should
          likely become a class.
    """
    __LOG_TAG = "Milhoja TF Assembler"

    @staticmethod
    def from_milhoja_json(name, internal_call_graph, jsons_all,
                          bridge_json, grid_json, logger):
        """
        .. todo::
            * Load JSON files carefully and accounting for different versions
              as is done in TaskFunction.from_milhoja_json.
            * If the internal call graph includes subroutines from different
              operations, then we likely need multiple bridges.  These are
              really operations.
            * Log arguments as debug information

        :param name: Name of the task function
        :param internal_call_graph: Refer to documentation in constructor for
            same argument
        :param jsons_all: Dictionary that returns Milhoja-JSON format
            subroutine specification file for each subroutine in call graph
        :param bridge_json: HACK JUST TO GET THINGS WORKING FOR NOW
        :param grid_json: WRITE THIS
        :param logger: Logger derived from milhoja.AbcLogger
        """
        # if not isinstance(logger, AbcLogger):
        #     raise TypeError("Unknown logger type")
        msg = f"Loading {name} Milhoja-JSON files"
        logger.log(TaskFunctionAssembler.__LOG_TAG, msg, LOG_LEVEL_BASIC)

        if not Path(bridge_json).is_file():
            msg = f"{bridge_json} does not exist or is not a file"
            raise ValueError(msg)
        if not Path(grid_json).is_file():
            msg = f"{grid_json} does not exist or is not a file"
            raise ValueError(msg)

        specs_all = {}
        for node in internal_call_graph:
            if isinstance(node, str):
                subroutines_all = [node]
            else:
                subroutines_all = node

            for subroutine in subroutines_all:
                filename = jsons_all[subroutine]
                if not Path(filename).is_file():
                    msg = f"{filename} does not exist or is not a file"
                    raise ValueError(msg)
                with open(filename, "r") as fptr:
                    specs_all[subroutine] = json.load(fptr)

        operation_name = Path(bridge_json).stem.lower()
        with open(bridge_json, "r") as fptr:
            bridge = json.load(fptr)

        with open(grid_json, "r") as fptr:
            grid_spec = json.load(fptr)

        return TaskFunctionAssembler(name, internal_call_graph, specs_all,
                                     bridge, operation_name, grid_spec,
                                     logger)

    def __init__(self, name, internal_call_graph, subroutine_specs_all,
                 bridge, operation_name, grid_spec, logger):
        """
        It is intended that users instantiate assemblers using the from_*
        classmethods.

        .. todo::
            * Error check graph and each specification here with informative
              error messages
            * Log arguments as debug information

        :param name: Name of the task function
        :param internal_call_graph: Refer to documentation in constructor for
            same argument
        :param subroutine_specs_all: Dictionary that returns subroutine
            specification as a Milhoja internal data structure for each
            subroutine in call graph
        :param bridge: WRITE THIS!
        :param operation_name: WRITE THIS!
        :param grid_spec: WRITE THIS!
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
        self.__subroutine_specs_all = subroutine_specs_all
        self.__bridge = bridge
        self.__operation_name = operation_name
        self.__grid_spec = grid_spec

        self.__dummies, self.__dummy_specs, self.__dummy_to_actuals = \
            self.__determine_unique_dummies(
                self.__bridge, self.__operation_name
            )

        self.__sanity_check_grid_spec()

        # ----- SANITY CHECK GRID SPECIFICATION
    def __sanity_check_grid_spec(self):
        """
        If this does not raise an error, then the specification is acceptable.

        .. todo::
            * This should also check types
        """
        expected = {"dimension", "nxb", "nyb", "nzb", "nguardcells"}
        actual = set(self.__grid_spec)
        if actual != expected:
            msg = f"Invalid set of grid specification keys ({actual})"
            raise ValueError(msg)

        dimension = self.__grid_spec["dimension"]
        if dimension not in [1, 2, 3]:
            msg = f"Invalid grid dimension ({dimension})"
            raise ValueError(msg)

        nxb = self.__grid_spec["nxb"]
        if nxb <= 0:
            raise ValueError(f"Non-positive NXB ({nxb})")

        nyb = self.__grid_spec["nyb"]
        if nyb <= 0:
            raise ValueError(f"Non-positive NYB ({nyb})")
        elif (dimension == 1) and (nyb != 1):
            raise ValueError(f"nyb > 1 for {dimension}D problem")

        nzb = self.__grid_spec["nzb"]
        if nzb <= 0:
            raise ValueError(f"Non-positive NZB ({nzb})")
        elif (dimension < 3) and (nzb != 1):
            raise ValueError(f"nzb > 1 for {dimension}D problem")

        n_gc = self.__grid_spec["nguardcells"]
        if n_gc < 0:
            raise ValueError(f"Negative N guardcells ({n_gc})")

    def __determine_unique_dummies(self, bridge, operation_name):
        """
        This is the workhorse of the assembler that identifies the minimal set
        of dummy arguments for the task function, writes the specifications for
        each dummy, and determines the mapping of task function dummy argument
        onto actual argument for each subroutine in the internal subroutine
        graph.

        .. todo::
            * What about lbound arguments!?
            * Milhoja should have an internal parser that can figure out the
              R/W/RW status of each variable of the grid data arrays in each
              subroutine in the internal call graph.  That information can be
              used to determine the variable masks for each grid_data
              structure.
            * Where to get unit and operation names from to compose unique
              scratch variable names?
            * Very low-priority optimization is determine if a scratch variable
              for one subroutine can double as scratch variable for another
              subroutine so that we can limit amount of scratch needed by TF.
        """
        tf_dummy_list = []
        tf_dummy_spec = {}
        dummy_to_actuals = {}

        # While the actual argument ordering likely does not matter.  Make sure
        # that our ordering is fixed so that tests will work on all platforms
        # and all versions of python.
        for arg_specs, arg_mappings in [
            self.__get_external(bridge, operation_name),
            self.__get_tile_metadata(), self.__get_grid_data(),
            self.__get_scratch(bridge, operation_name)
        ]:
            assert set(arg_specs).intersection(tf_dummy_spec) == set()
            tf_dummy_spec.update(arg_specs)
            assert set(arg_mappings).intersection(dummy_to_actuals) == set()
            dummy_to_actuals.update(arg_mappings)
            assert set(arg_specs).intersection(tf_dummy_list) == set()
            tf_dummy_list += sorted(arg_specs.keys())

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
            the subroutines in that node can be run concurrently.
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
        return self.__subroutine_specs_all[subroutine]

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

    def __get_external(self, bridge, operation_name):
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
        bridge_spec = bridge["external"]

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
                        bridge_name = arg_spec["name"].strip()
                        assert bridge_name.startswith("_")
                        tf_dummy = f"{operation_name}{bridge_name}"
                        if tf_dummy not in tf_dummy_spec:
                            tmp_spec = bridge_spec[bridge_name].copy()
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
                            raise NotImplementedError("Test case!")

                        if tf_dummy not in tf_dummy_spec:
                            tf_dummy_spec[tf_dummy] = arg_spec
                            assert tf_dummy not in dummy_to_actuals
                            dummy_to_actuals[tf_dummy] = []
                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        return tf_dummy_spec, dummy_to_actuals

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
            This presently supports only one grid data structure per index
            space

        :return: tf_dummy_spec, dummy_to_actuals where
            * tf_dummy_spec is the specification of grid data dummy arguments
              for the task function and
            * dummy_to_actuals is the mapping
        """
        VARIABLES_IN = "variables_in"
        VARIABLES_OUT = "variables_out"

        # Specify unique, consistent names for TF dummy arguments
        spaces_mapping = {
            "center": "CC_{0}",
            "fluxx": "FLX_{0}",
            "fluxy": "FLY_{0}",
            "fluxz": "FLZ_{0}"
        }

        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    # Get copy since we will be updating specs below
                    arg_spec = spec["argument_specifications"][arg].copy()
                    source = arg_spec["source"]
                    if source in TaskFunction.GRID_DATA_ARGUMENT:
                        index_space, grid_index = arg_spec["structure_index"]
                        assert index_space.lower() in spaces_mapping
                        if grid_index != 1:
                            msg = "Only one {} data structure presently allowed"
                            raise NotImplementedError(msg.format(index_space))

                        # Update variable masking
                        vars_in = []
                        vars_out = []
                        if "R" in arg_spec:
                            vars_in = arg_spec["R"]
                            del arg_spec["R"]
                        if "W" in arg_spec:
                            vars_out = arg_spec["W"]
                            del arg_spec["W"]
                        if "RW" in arg_spec:
                            rw_idx = arg_spec["RW"]
                            del arg_spec["RW"]
                            vars_in += rw_idx
                            vars_out += rw_idx

                        space = index_space.lower()
                        tf_dummy = spaces_mapping[space].format(grid_index)
                        if tf_dummy not in tf_dummy_spec:
                            if vars_in:
                                min_in = min(vars_in)
                                max_in = max(vars_in)
                                arg_spec[VARIABLES_IN] = [min_in, max_in]
                            if vars_out:
                                min_out = min(vars_out)
                                max_out = max(vars_out)
                                arg_spec[VARIABLES_OUT] = [min_out, max_out]
                            tf_dummy_spec[tf_dummy] = arg_spec
                            assert tf_dummy not in dummy_to_actuals
                            dummy_to_actuals[tf_dummy] = []
                        else:
                            tmp_spec = tf_dummy_spec[tf_dummy].copy()
                            if vars_in:
                                min_in = min(vars_in)
                                max_in = max(vars_in)
                                if VARIABLES_IN not in arg_spec:
                                    tmp_spec[VARIABLES_IN] = [min_in, max_in]
                                else:
                                    vars_in = arg_spec[VARIABLES_IN]
                                    tmp_spec[VARIABLES_IN] = [
                                        min(vars_in[0], min_in),
                                        max(vars_in[1], max_in)
                                    ]
                            if vars_out:
                                min_out = min(vars_out)
                                max_out = max(vars_out)
                                if VARIABLES_OUT not in arg_spec:
                                    tmp_spec[VARIABLES_OUT] = [min_out, max_out]
                                else:
                                    vars_out = arg_spec[VARIABLES_OUT]
                                    tmp_spec[VARIABLES_OUT] = [
                                        min(vars_out[0], min_out),
                                        max(vars_out[1], max_out)
                                    ]
                            tf_dummy_spec[tf_dummy] = tmp_spec

                        actual = (subroutine, arg, idx+1)
                        dummy_to_actuals[tf_dummy].append(actual)

        return tf_dummy_spec, dummy_to_actuals

    def __get_scratch(self, bridge, operation_name):
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
        bridge_spec = bridge["scratch"]

        tf_dummy_spec = {}
        dummy_to_actuals = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx, arg in enumerate(spec["argument_list"]):
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source == TaskFunction.SCRATCH_ARGUMENT:
                        bridge_name = arg_spec["name"].strip()
                        assert bridge_name.startswith("_")
                        tf_dummy = f"{operation_name}{bridge_name}"
                        if tf_dummy not in tf_dummy_spec:
                            tmp_spec = bridge_spec[bridge_name].copy()
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
            if spec["source"] in TaskFunction.EXTERNAL_ARGUMENT:
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
            if spec["source"] in TaskFunction.GRID_DATA_ARGUMENT:
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
            if spec["source"] in TaskFunction.SCRATCH_ARGUMENT:
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

        # ----- TASK FUNCTION
        tf_spec = spec["task_function"]
        expected = {"language", "processor",
                    "cpp_header", "cpp_source",
                    "c2f_source", "fortran_source"}
        actual = set(tf_spec)
        if actual != expected:
            msg = f"Invalid TF specification keys ({actual})"
            raise ValueError(msg)

        language = tf_spec["language"]
        if language.lower() not in ["c++", "fortran"]:
            raise ValueError(f"Unsupported TF language ({language})")

        processor = tf_spec["processor"]
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
        data_item = spec["data_item"]
        expected = {"type", "byte_alignment", "header", "source"}
        actual = set(data_item)
        if actual != expected:
            msg = f"Invalid data item specification keys ({actual})"
            raise ValueError(msg)

        item_type = data_item["type"]
        if item_type.lower() not in ["tile", "datapacket"]:
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
        spec[group] = self.__grid_spec

        # ----- INCLUDE TASK FUNCTION SPECIFICATION
        group = "task_function"
        assert group not in spec
        spec[group] = tf_spec[group]
        spec[group]["name"] = self.task_function_name
        spec[group]["argument_list"] = self.dummy_arguments

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
