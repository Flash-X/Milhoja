import json

from pathlib import Path
from collections import OrderedDict

from . import MILHOJA_JSON_FORMAT
from . import CURRENT_MILHOJA_JSON_VERSION
from . import TaskFunction


class TaskFunctionAssembler(object):
    """
    .. todo::
        * Write documentation here and for all methods in class.
        * internal_call_graph is used here and in TaskFunction.  It should
          likely become a class.
        * Add in logging at different verbosity levels?
    """

    @staticmethod
    def from_milhoja_json(name, internal_call_graph, jsons_all):
        """
        .. todo::
            * Load JSON files carefully and accounting for different versions
              as is done in TaskFunction.from_milhoja_json.

        :param name: Name of the task function
        :param internal_call_graph: Refer to documentation in constructor for
            same argument
        :param jsons_all: Dictionary that returns Milhoja-JSON format
            subroutine specification file for each subroutine in call graph
        """
        specs_all = {}
        for node in internal_call_graph:
            if isinstance(node, str):
                subroutines_all = [node]
            else:
                subroutines_all = node

            for subroutine in subroutines_all:
                with open(jsons_all[subroutine], "r") as fptr:
                    specs_all[subroutine] = json.load(fptr)

        return TaskFunctionAssembler(name, internal_call_graph, specs_all)

    def __init__(self, name, internal_call_graph, subroutine_specs_all):
        """
        It is intended that users instantiate assemblers using the from_*
        classmethods.

        .. todo::
            * Error check graph and each specification here with informative
              error messages

        :param name: Name of the task function
        :param internal_call_graph: Refer to documentation in constructor for
            same argument
        :param subroutine_specs_all: Dictionary that returns subroutine
            specification as a Milhoja internal data structure for each
            subroutine in call graph
        """
        super().__init__()

        self.__tf_name = name
        self.__call_graph = internal_call_graph
        self.__subroutine_specs_all = subroutine_specs_all

        self.__dummies, self.__dummy_specs, self.__arg_mapping = \
            self.__determine_unique_dummies()

    def __determine_unique_dummies(self):
        """
        This is the workhorse of the assembler that identifies the dummy
        arguments for the task function, writes the specifications for each,
        and determines the mapping of task function dummy argument onto actual
        argument for each subroutine in the internal subroutine graph.

        .. todo::
            * This is written so that a single, particular test will pass.  It
              needs to be written for real.  Identification of unique tile
              metadata and grid data structures should be simple.  How to
              identify if external variables across all subroutines are the
              same?  Ditto for scratch?
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
            * At the moment, tile metadata is evaluated as already being inside 
              of the json if the argument specification is the exact same as a 
              different argument specification that has already been inserted into 
              the json. A small optimization would be to only check subsets of 
              the information in the arg spec. For example, there might be 2 
              tile_coordinate sources that use different bounds. This could be 
              collapsed into one tile_coordinate variable.
        """
        # need tfspecs,
        # argument mapping,
        # and argument list based on all tf jsons.
        full_specs = OrderedDict()
        mapping = {}
        arguments = []

        # While the actual argument ordering likely does not matter.  Make sure
        # that our ordering is fixed so that tests will work on all platforms
        # and all versions of python.

        for arg_specs,arg_mappings in [
            self.__get_external, self.__get_tile_metadata, 
            self.__get_grid_data, self.__get_scratch
        ]:
            full_specs.update(arg_specs)
            mapping.update(arg_mappings)

        arguments = list(full_specs.keys())
        # print("Arguments: ", arguments)
        # print("Argument specs: ", json.dumps(full_specs, indent=4))
        # print("Argument mapping: ", json.dumps(mapping, indent=4))

        return arguments, full_specs, mapping

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
    
    @property
    def __get_external(self):
        needed = OrderedDict()
        mapping = OrderedDict()
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx,arg in enumerate(spec["argument_list"]):
                    source = spec["argument_specifications"][arg]["source"]
                    source_name = spec["argument_specifications"][arg].get("source_name", None)
                    if source == TaskFunction.EXTERNAL_ARGUMENT:
                        key = source_name
                        if not key:
                            key = f"{subroutine}_{arg}"
                        else:
                            del spec["argument_specifications"][arg]["source_name"] # delete source name?
                        if key not in needed:
                            needed[key] = spec["argument_specifications"][arg]
                        if key not in mapping:
                            mapping[key] = []
                        mapping[key].append((subroutine, arg, idx+1))
        return needed,mapping
    
    @property
    def __get_tile_metadata(self):
        
        # def sort_key(value):
        #     # order = ["tile_deltas", "tile_ho", "tile_hi"]
        #     return order.index(value[0])

        needed = OrderedDict()
        mapping = OrderedDict()
        source_arg_mapping = {}
        # TODO: Wow this is ugly
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx,arg in enumerate(spec["argument_list"]):
                    source = spec["argument_specifications"][arg]["source"]
                    arg_spec = spec["argument_specifications"][arg]
                    mapping_key = source if len(arg_spec) == 1 else arg
                    if source in TaskFunction.TILE_METADATA_ALL:
                        if source in source_arg_mapping:
                            for variable in source_arg_mapping[source]:
                                var_spec = needed[variable]
                                if var_spec == arg_spec:
                                    mapping_key = variable
                                    break
                            if mapping_key not in needed:
                                needed[mapping_key] = arg_spec
                            source_arg_mapping[source].add(mapping_key)
                        else:
                            needed[mapping_key] = arg_spec
                            source_arg_mapping[source] = {mapping_key}
                        if mapping_key not in mapping:
                            mapping[mapping_key] = []
                        mapping[mapping_key].append((subroutine, arg, idx+1))

        return OrderedDict(sorted(needed.items())),OrderedDict(sorted(mapping.items()))

    @property
    def __get_grid_data(self):
        spaces_mapping = {
            "center": "CC_{0}",
            "fluxx": "FLX_{0}",
            "fluxy": "FLY_{0}",
            "fluxz": "FLZ_{0}"
        }

        needed = OrderedDict()
        mapping = OrderedDict()
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx,arg in enumerate(spec["argument_list"]):
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source in TaskFunction.GRID_DATA_ARGUMENT:
                        index_space, grid_index = arg_spec["structure_index"]
                        assert index_space.lower() in spaces_mapping
                        assert grid_index > 0 # should it be 0?
                        name = spaces_mapping[index_space.lower()].format(grid_index)
                        if name not in needed:
                            needed[name] = arg_spec
                        # TODO: We need to union the variable masking!
                        # update mapping
                        if name not in mapping:
                            mapping[name] = []
                        mapping[name].append((subroutine, arg, idx+1))
        return needed,mapping

    @property
    def __get_scratch(self):
        scratch_name_mapping = {
            "auxc": "hydro_op1_auxc",
            "flx": "hydro_op1_flX",
            "fly": "hydro_op1_flY",
            "flz": "hydro_op1_flZ"
        }
        needed = OrderedDict()
        mapping = OrderedDict()
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for idx,arg in enumerate(spec["argument_list"]):
                    source = spec["argument_specifications"][arg]["source"]
                    source_name = spec["argument_specifications"][arg].get("source_name", None)
                    if source == TaskFunction.SCRATCH_ARGUMENT:
                        key = source_name
                        if not key:
                            key = f"{subroutine}_{arg}"
                        else:
                            del spec["argument_specifications"][arg]["source_name"] # delete source name?
                        key = scratch_name_mapping.get(key.lower(), key)
                        if key not in needed:
                            needed[key] = spec["argument_specifications"][arg]
                        if key not in mapping:
                            mapping[key] = []
                        mapping[key].append((subroutine, arg, idx+1))
        return needed,mapping

    @property
    def tile_metadata_arguments(self):
        """
        .. todo::
            * This is presently used by __determine_unique_dummies.  That
              helper should determined all unique dummies by itself and this
              should just use the set using the results.
            * Is this needed?

        :return: Dict of task function's dummy arguments that are classified as
            tile metatdata and their arg specs
        """
        needed = set()
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for arg in spec["argument_list"]:
                    source = spec["argument_specifications"][arg]["source"]
                    if source in TaskFunction.TILE_METADATA_ALL:
                        needed = needed.union(set([source]))
        return needed


    @property
    def external_arguments(self):
        """
        .. todo::
            * This is presently used by __determine_unique_dummies.  That
              helper should determine all unique dummies by itself and this
              should just use the set using the results.
            * Is this needed?

        :return: Set of task function's dummy arguments that are classified as
            external arguments
        """
        needed = set()
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for arg in spec["argument_list"]:
                    source = spec["argument_specifications"][arg]["source"]
                    if source == TaskFunction.EXTERNAL_ARGUMENT:
                        needed = needed.union(set([arg]))
        return needed


    @property
    def grid_data_structures(self):
        """
        .. todo::
            * This is presently used by __determine_unique_dummies.  That
              helper should determined all unique dummies by itself and this
              should just use the set using the results.
            * Is this needed?
            * We have tests in the package that use FLUX[XYZ].  Use those to
              expand out the structures that this can work with.

        :return: Set of grid data structures whose data are read from or set by
            the task function
        """
        needed = {}
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for arg in spec["argument_list"]:
                    arg_spec = spec["argument_specifications"][arg]
                    source = arg_spec["source"]
                    if source in TaskFunction.GRID_DATA_ARGUMENT:
                        index_space, grid_index = arg_spec["structure_index"]
                        assert index_space.lower() == "center"
                        assert grid_index == 1
                        if index_space not in needed:
                            needed[index_space] = set([grid_index])
                        else:
                            needed[index_space] = \
                                needed[index_space].union(set([grid_index]))
        return needed


    @property
    def scratch_arguments(self):
        """
        .. todo::
            * This is presently used by __determine_unique_dummies.  That
              helper should determined all unique dummies by itself and this
              should just use the set using the results.
            * Is this needed?

        :return: Set of task function's dummy arguments that are classified as
            scratch arguments
        """
        scratch_name_mapping = {
            "auxc": "hydro_op1_auxc",
            "flx": "hydro_op1_flX",
            "fly": "hydro_op1_flY",
            "flz": "hydro_op1_flZ"
        }

        needed = set()
        for node in self.internal_subroutine_graph:
            for subroutine in node:
                spec = self.subroutine_specification(subroutine)
                for arg in spec["argument_list"]:
                    source = spec["argument_specifications"][arg]["source"]
                    if source == TaskFunction.SCRATCH_ARGUMENT:
                        needed = needed.union(set([arg]))
        return needed


    def to_milhoja_json(self, filename, overwrite):
        """
        Write the assembled task function to the given file using the current
        version of the Milhoja-JSON task function specification format.

        .. todo::
            * How do we get the grid information?
            * How to get general task function information?
            * How do we get data item information?
            * Milhoja should have an internal parser that gets the argument
              list for each subroutine in the internal call graph.  Then the
              given subroutine JSON files don't need to specify that.

        :param filename: Name and full path of file to write to
        :param overwrite: Raise exception if the file exists and this is True
        """
        spec = {}
        spec["format"] = [MILHOJA_JSON_FORMAT, CURRENT_MILHOJA_JSON_VERSION]

        # ----- INCLUDE GRID SPECIFICATION
        group = "grid"
        assert group not in spec
        spec[group] = {}
        spec[group]["dimension"] = 3
        spec[group]["nxb"] = 16
        spec[group]["nyb"] = 16
        spec[group]["nzb"] = 16
        spec[group]["nguardcells"] = 1

        # ----- INCLUDE TASK FUNCTION SPECIFICATION
        group = "task_function"
        assert group not in spec
        spec[group] = {}
        spec[group]["name"] = self.task_function_name
        spec[group]["language"] = "Fortran"
        spec[group]["processor"] = "GPU"
        spec[group]["cpp_header"] = "gpu_tf_hydro_Cpp2C.h"
        spec[group]["cpp_source"] = "gpu_tf_hydro_Cpp2C.cpp"
        spec[group]["c2f_source"] = "gpu_tf_hydro_C2F.F90"
        spec[group]["fortran_source"] = "gpu_tf_hydro.F90"
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
        spec[group] = {}
        spec[group]["type"] = "DataPacket"
        spec[group]["byte_alignment"] = 16
        spec[group]["header"] = "DataPacket_gpu_tf_hydro.h"
        spec[group]["source"] = "DataPacket_gpu_tf_hydro.cpp"

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
                    for key, value in self.__arg_mapping.items():
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

        print(json.dumps(spec, indent=4))

        if (not overwrite) and Path(filename).exists():
            raise RuntimeError(f"{filename} already exists")

        with open(filename, "w") as fptr:
            json.dump(
                spec, fptr, ensure_ascii=True, allow_nan=False, indent="\t"
            )
