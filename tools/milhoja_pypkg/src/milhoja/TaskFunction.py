import json

from pathlib import Path

class TaskFunction(object):
    """
    """
    __JSON_FORMAT_NAME = "Milhoja-native JSON"
    __CURRENT_JSON_VERSION = "1.0.0"

    @staticmethod
    def from_json(filename):
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
        if format_name.lower() != TaskFunction.__JSON_FORMAT_NAME.lower():
            raise ValueError(f"Unknown JSON format {format_name}")

        # Only one version of Milhoja-native JSON format.
        # Therefore, contents already in Milhoja-internal format.
        if version.lower() != TaskFunction.__CURRENT_JSON_VERSION.lower():
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
        self.__device_spec = specification["device"]
        self.__subroutine_spec = specification["subroutines"]

        # ----- ERROR CHECK ARGUMENTS
        if not self.__filename.is_file():
            raise ValueError(f"{filename} does not exist")

        # ----- ERROR CHECK CONTENTS

    @property
    def specification_filename(self):
        return self.__filename

    @property
    def name(self):
        return self.__tf_spec["name"]

    @property
    def language(self):
        return self.__tf_spec["language"]

    @property
    def argument_list(self):
        """
        Returned in the proper order
        """
        return self.__tf_spec["argument_list"]

    def argument_specification(self, argument):
        """
        """
        return self.__tf_spec["argument_specifications"][argument]

    @property
    def device_information(self):
        """
        """
        device = self.__device_spec["hardware"]

        if device.lower() == "cpu":
            return {"device": "CPU"}
        elif device.lower() == "gpu":
            return {"device": "GPU", \
                    "byte_alignment": self.__device_spec["byte_alignment"]}

        raise ValueError(f"Unable to offload to {device}")

    @property
    def tile_metadata(self):
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

#    @property
#    def external(self):
#
#    @property
#    def scratch(self):
#
#    @property
#    def tile_in(self):
#
#    @property
#    def tile_in_out(self):
#
#    @property
#    def tile_out(self):

    def to_json(self, filename):
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
