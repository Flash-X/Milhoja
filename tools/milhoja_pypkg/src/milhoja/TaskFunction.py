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
        if format_name != TaskFunction.__JSON_FORMAT_NAME:
            raise ValueError(f"Unknown JSON format {format_name}")

        # Only one version of Milhoja-native JSON format.
        # Therefore, contents already in Milhoja-internal format.
        if version != TaskFunction.__CURRENT_JSON_VERSION:
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
        self.__tf_spec = specification

        # ----- ERROR CHECK ARGUMENTS
        if not self.__filename.is_file():
            raise ValueError(f"{filename} does not exist")

        format_name, version = self.__tf_spec["format"]
        format_name = format_name.lower()
        version = version.lower()
        if format_name != TaskFunction.__JSON_FORMAT_NAME.lower():
            raise ValueError(f"Invalid configuration format {format_name}")
        elif version != TaskFunction.__CURRENT_JSON_VERSION.lower():
            raise ValueError(f"Invalid configuration version {version}")

        # ----- ERROR CHECK CONTENTS

    @property
    def filename(self):
        return self.__filename

    @property
    def tile_metadata(self):
        """
        """
        # TODO: Should this be constants so that the list is publicly
        # available?
        KEYS = ["tile_gridIndex",
                "tile_level",
                "tile_lo", "tile_hi",
                "tile_lbound", "tile_ubound",
                "tile_deltas",
                "tile_coordinates",
                "tile_cellVolumes"]

        metadata_specs_all = {}

        arg_specs_all = self.__tf_spec["argument_specifications"]
        for arg in self.__tf_spec["argument_list"]:
            arg_spec = arg_specs_all[arg]
            if arg_spec["source"] in KEYS:
                assert arg not in metadata_specs_all
                metadata_specs_all[arg] = arg_spec

        return metadata_specs_all 

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
