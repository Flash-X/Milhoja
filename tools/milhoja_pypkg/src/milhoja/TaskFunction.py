import json

from pathlib import Path

from . import MILHOJA_JSON_FORMAT
from . import CURRENT_MILHOJA_JSON_VERSION


class TaskFunction(object):
    """
    """
    # Main keys into dictionaries returned by output_filenames.
    CPP_TF_KEY = "cpp_tf"
    C2F_KEY = "c2f"
    FORTRAN_TF_KEY = "fortran_tf"
    DATA_ITEM_KEY = "data_item"

    # Case-sensitive source keys to be used in specification files (e.g.,
    # Milhoja-JSON files), here, and in code generators.
    #
    # Scheme for keys
    # - concise one word all lowercase for all keys when possible
    # - tile metadata keys all begin with tile_ (all lowercase) with remainder
    #   in camelcase with no separation
    EXTERNAL_ARGUMENT = "external"
    SCRATCH_ARGUMENT = "scratch"
    GRID_DATA_ARGUMENT = "grid_data"
    TILE_GRID_INDEX = "tile_gridIndex"
    TILE_LEVEL = "tile_level"
    TILE_LO = "tile_lo"
    TILE_HI = "tile_hi"
    TILE_LBOUND = "tile_lbound"
    TILE_UBOUND = "tile_ubound"
    TILE_DELTAS = "tile_deltas"
    TILE_COORDINATES = "tile_coordinates"
    TILE_FACE_AREAS = "tile_faceAreas"
    TILE_CELL_VOLUMES = "tile_cellVolumes"

    @staticmethod
    def from_milhoja_json(filename):
        """
        """
        # ----- ERROR CHECK ARGUMENTS
        fname = Path(filename).resolve()
        if not fname.is_file():
            raise ValueError(f"{filename} does not exist or is not a file")

        # ----- LOAD CONFIGURATION
        with open(fname, "r") as fptr:
            configuration = json.load(fptr)

        # ----- CONVERT TO CURRENT MILHOJA INTERNAL TASK FUNCTION
        #       REPRESENTATION
        format_name, version = configuration["format"]

        # Only one JSON internal format presently
        if format_name.lower() != MILHOJA_JSON_FORMAT.lower():
            raise ValueError(f"Unknown JSON format {format_name}")

        # Only one version of Milhoja-native JSON format.
        # Therefore, contents already in Milhoja-internal format.
        if version.lower() != CURRENT_MILHOJA_JSON_VERSION.lower():
            raise ValueError(f"Unknown {format_name} version v{version}")

        return TaskFunction(filename, configuration)

    def __init__(self, filename, specification):
        """
        The constructor eagerly sanity checks the full contents of the packet
        so that error checking does not depend on what member functions are
        called and all other member functions can assume correctness.

        .. todo::
            * Perform full sanity check of specification
            * Should correction do automatic fixing of bad case when it does
              check so that all code generators can be written in
              case-sensitive way?

        :param specification: Internal Milhoja representation
        """
        super().__init__()

        self.__filename = Path(filename).resolve()
        self.__spec = specification
        self.__tf_spec = specification["task_function"]
        self.__data_spec = specification["data_item"]
        self.__subroutine_spec = specification["subroutines"]
        self.__grid_spec = specification["grid"]

    @property
    def specification_filename(self):
        return self.__filename

    @property
    def specification_format(self):
        return self.__spec["format"]

    @property
    def output_filenames(self):
        """
        :return: Dictionary of filenames where value is a dictionary
        consisting of the key "source" and potentially "header".
        """
        cpp_tf_hdr = self.__tf_spec["cpp_header"].strip()
        cpp_tf_src = self.__tf_spec["cpp_source"].strip()
        c2f_src = self.__tf_spec["c2f_source"].strip()
        fortran_tf_src = self.__tf_spec["fortran_source"].strip()
        data_item_hdr = self.__data_spec["header"].strip()
        data_item_src = self.__data_spec["source"].strip()

        assert cpp_tf_hdr != ""
        assert cpp_tf_src != ""
        assert data_item_hdr != ""
        assert data_item_src != ""

        filenames = {}
        filenames[TaskFunction.DATA_ITEM_KEY] = \
            {"header": data_item_hdr, "source": data_item_src}
        filenames[TaskFunction.CPP_TF_KEY] = \
            {"header": cpp_tf_hdr, "source": cpp_tf_src}

        processor = self.processor
        language = self.language
        if processor.lower() == "cpu" and language.lower() == "c++":
            assert c2f_src == ""
            assert fortran_tf_src == ""
        elif processor.lower() == "gpu" and language.lower() == "fortran":
            assert c2f_src != ""
            assert fortran_tf_src != ""

            filenames[TaskFunction.C2F_KEY] = {"source": c2f_src}
            filenames[TaskFunction.FORTRAN_TF_KEY] = {"source": fortran_tf_src}
        else:
            raise NotImplementedError("Waiting for test cases")

        return filenames

    @property
    def name(self):
        return self.__tf_spec["name"]

    @property
    def language(self):
        return self.__tf_spec["language"]

    @property
    def processor(self):
        return self.__tf_spec["processor"]

    @property
    def data_item(self):
        return self.__data_spec["type"]

    @property
    def data_item_class_name(self):
        if self.data_item.lower() == "tilewrapper":
            return f"Tile_{self.name}"
        raise NotImplementedError("Only setup for TileWrapper right now")

    @property
    def grid_dimension(self):
        return self.__grid_spec["dimension"]

    @property
    def block_interior_shape(self):
        shape = (
            self.__grid_spec["nxb"],
            self.__grid_spec["nyb"],
            self.__grid_spec["nzb"]
        )
        return shape

    @property
    def n_guardcells(self):
        return self.__grid_spec["nguardcells"]

    @property
    def dummy_arguments(self):
        """
        Returned in the proper order
        """
        return self.__tf_spec["argument_list"]

    def argument_specification(self, argument):
        """
        .. todo::
            This should reinterpret variable types so that we return the
            correct type for the task function's processor.
        """
        spec = self.__tf_spec["argument_specifications"][argument]

        src_to_adjust = [
            TaskFunction.EXTERNAL_ARGUMENT, TaskFunction.SCRATCH_ARGUMENT
        ]
        if ((spec["source"].lower() in src_to_adjust) and
                (spec["type"].lower() == "real") and
                (self.processor.lower() == "cpu")):
            spec["type"] = "milhoja::Real"

        return spec

    @property
    def constructor_dummy_arguments(self):
        """
        """
        # We want this to always generate the same argument order
        arguments = []
        for arg in self.dummy_arguments:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == TaskFunction.EXTERNAL_ARGUMENT:
                arguments.append((arg, arg_spec["type"]))

        return arguments

    @property
    def tile_metadata_arguments(self):
        """
        """
        KEYS_ALL = [
            TaskFunction.TILE_GRID_INDEX,
            TaskFunction.TILE_LEVEL,
            TaskFunction.TILE_LO, TaskFunction.TILE_HI,
            TaskFunction.TILE_LBOUND, TaskFunction.TILE_UBOUND,
            TaskFunction.TILE_DELTAS,
            TaskFunction.TILE_COORDINATES,
            TaskFunction.TILE_FACE_AREAS,
            TaskFunction.TILE_CELL_VOLUMES
        ]

        metadata_all = {}
        for arg in self.dummy_arguments:
            arg_spec = self.argument_specification(arg)
            key = arg_spec["source"]
            if key in KEYS_ALL:
                if key not in metadata_all:
                    metadata_all[key] = [arg]
                else:
                    metadata_all[key].append(arg)

        return metadata_all

    @property
    def external_arguments(self):
        """
        """
        external_all = set()
        for arg in self.dummy_arguments:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == TaskFunction.EXTERNAL_ARGUMENT:
                assert arg not in external_all
                external_all.add(arg)

        return external_all

    @property
    def scratch_arguments(self):
        """
        This is only scratch arguments explicitly requested in the
        specification.  In particular, it does not include scratch variables
        needed internally by Milhoja.
        """
        scratch_all = set()
        for arg in self.dummy_arguments:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == TaskFunction.SCRATCH_ARGUMENT:
                assert arg not in scratch_all
                scratch_all.add(arg)

        return scratch_all

    @property
    def tile_in_arguments(self):
        """
        """
        data_all = set()
        for arg in self.dummy_arguments:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == TaskFunction.GRID_DATA_ARGUMENT:
                has_in = ("variables_in" in arg_spec)
                has_out = ("variables_out" in arg_spec)
                if has_in and (not has_out):
                    assert arg not in data_all
                    data_all.add(arg)

        return data_all

    @property
    def tile_in_out_arguments(self):
        """
        """
        data_all = set()
        for arg in self.dummy_arguments:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == TaskFunction.GRID_DATA_ARGUMENT:
                has_in = ("variables_in" in arg_spec)
                has_out = ("variables_out" in arg_spec)
                if has_in and has_out:
                    assert arg not in data_all
                    data_all.add(arg)

        return data_all

    @property
    def tile_out_arguments(self):
        """
        """
        data_all = set()
        for arg in self.dummy_arguments:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == TaskFunction.GRID_DATA_ARGUMENT:
                has_in = ("variables_in" in arg_spec)
                has_out = ("variables_out" in arg_spec)
                if (not has_in) and has_out:
                    assert arg not in data_all
                    data_all.add(arg)

        return data_all

    @property
    def internal_subroutine_graph(self):
        """
        :return: Generator for iterating in correct order over the nodes in the
            internal subroutine graph of the task function.  Each node contains
            one or more subroutines.  If more than one, it is understood that
            the subroutines in that node can be run concurrently.
        """
        for node in self.__tf_spec["subroutine_call_graph"]:
            if isinstance(node, str):
                yield [node]
            else:
                yield node

    def subroutine_interface_file(self, subroutine):
        """
        """
        return self.__subroutine_spec[subroutine]["interface_file"]

    def subroutine_dummy_arguments(self, subroutine):
        """
        """
        return self.__subroutine_spec[subroutine]["argument_list"]

    def subroutine_actual_arguments(self, subroutine):
        """
        """
        dummies = self.subroutine_dummy_arguments(subroutine)
        mapping = self.__subroutine_spec[subroutine]["argument_mapping"]
        return [mapping[dummy] for dummy in dummies]

    @property
    def n_streams(self):
        """
        Raises an error if task function's data item does not use streams.

        :return: Maximum number of streams required during execution of a
            DataPacket-based tasked function.  This is the number of internal
            subroutines that could be running concurrently during execution of
            the task function.
        """
        if self.data_item.lower() == "tilewrapper":
            raise ValueError("Streams are not used with TileWrappers")
        elif self.data_item.lower() == "datapacket":
            n_streams = -1
            for node in self.internal_subroutine_graph:
                n_streams = max([n_streams, len(node)])
            return n_streams

        raise ValueError(f"Unknown data item type {self.data_item}")

    def to_milhoja_json(self, filename):
        """
        Write the object's full configuration to a Milhoja-native JSON file
        using the current version of the JSON format.
        """
        # ----- ERROR CHECK ARGUMENTS
        fname = Path(filename).resolve()
        if fname.exists():
            raise ValueError(f"{filename} already exists")

        with open(fname, "w") as fptr:
            json.dump(self.__config, fptr)
