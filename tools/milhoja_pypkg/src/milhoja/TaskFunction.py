import json

from pathlib import Path

from . import MILHOJA_JSON_FORMAT
from . import CURRENT_MILHOJA_JSON_VERSION

class TaskFunction(object):
    """
    """
    # Main keys into dictionaries returned by output_filenames.
    CPP_TF_KEY = "cpp_tf"
    DATA_ITEM_KEY = "data_item"

    @staticmethod
    def from_milhoja_json(filename):
        """
        """
        # ----- ERROR CHECK ARGUMENTS
        fname = Path(filename).resolve()
        if not fname.is_file():
            raise ValueError(f"{filename} does not exist")

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
            raise ValueError(f"Unknown Milhoja JSON version v{version}")

        return TaskFunction(filename, configuration)

    def __init__(self, filename, specification):
        """
        The constructor eagerly sanity checks the full contents of the packet
        so that error checking does not depend on what member functions are
        called and all other member functions can assume correctness.

        Parameters
            specification - internal Milhoja representation
        """
        super().__init__()

        self.__filename = Path(filename).resolve()
        self.__spec = specification
        self.__tf_spec = specification["task_function"]
        self.__data_spec = specification["data_item"]
        self.__subroutine_spec = specification["subroutines"]

        # ----- ERROR CHECK ARGUMENTS
        if not self.__filename.is_file():
            raise ValueError(f"{filename} does not exist")

        # ----- ERROR CHECK CONTENTS

    @property
    def specification_filename(self):
        return self.__filename

    @property
    def specification_format(self):
        return configuration["format"]

    @property
    def output_filenames(self):
        """
        """
        cpp_tf_hdr = self.__tf_spec["cpp_header"]
        cpp_tf_src = self.__tf_spec["cpp_source"]
        c2f_src = self.__tf_spec["c2f_source"]
        fortran_tf_src = self.__tf_spec["fortran_source"]
        data_item_hdr = self.__data_spec["header"]
        data_item_src = self.__data_spec["source"]

        assert cpp_tf_hdr != ""
        assert cpp_tf_src != ""
        assert data_item_hdr != ""
        assert data_item_src != ""

        filenames = {}
        filenames[TaskFunction.DATA_ITEM_KEY] = \
            {"header": data_item_hdr, "source": data_item_src}
        filenames[TaskFunction.CPP_TF_KEY] = \
            {"header": cpp_tf_hdr, "source": cpp_tf_src}

        # TODO: Each key should be associated with a dictionary that uses
        # "header" and "source" appropriately
        processor = self.processor
        language = self.language
        if processor.lower() == "cpu" and language.lower() == "c++":
            assert c2f_src == ""
            assert fortran_tf_src == ""
        else:
            raise NotImplementedError("Wait for Wesley")

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
    def argument_list(self):
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

        src_to_adjust = ["external", "scratch"]
        if      (spec["source"].lower() in src_to_adjust) \
            and (spec["type"].lower() == "real") \
            and (self.processor.lower() == "cpu"):
                spec["type"] = "milhoja::Real"

        return spec

    @property
    def constructor_argument_list(self):
        """
        """
        # We want this to always generate the same argument order
        arguments = []
        for arg in self.argument_list:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == "external":
                arguments.append((arg, arg_spec["type"]))

        return arguments

    @property
    def tile_metadata_arguments(self):
        """
        All keys are returned in full lowercase for case-insensitive
        comparisons by calling code.
        """
        # All keys in this list must be lowercase
        KEYS_ALL = ["tile_gridindex",
                    "tile_level",
                    "tile_lo", "tile_hi",
                    "tile_lbound", "tile_ubound",
                    "tile_deltas",
                    "tile_coordinates",
                    "tile_faceareas",
                    "tile_cellvolumes"]

        metadata_all = {}
        for arg in self.argument_list:
            arg_spec = self.argument_specification(arg)
            key = arg_spec["source"].lower()
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
        for arg in self.argument_list:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == "external":
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
        for arg in self.argument_list:
            arg_spec = self.argument_specification(arg)
            if arg_spec["source"].lower() == "scratch":
                assert arg not in scratch_all
                scratch_all.add(arg)

        return scratch_all

#    @property
#    def tile_in(self):
#
#    @property
#    def tile_in_out(self):
#
#    @property
#    def tile_out(self):

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
