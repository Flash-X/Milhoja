from pathlib import Path

from milhoja import (
    LOG_LEVEL_BASIC,
    TaskFunction,
    generate_data_item, generate_task_function
)


def generate_code(
            tf_spec_jsons_all, destination, overwrite, library_path,
            indent, makefile_filename,
            logger
        ):
    """
    Generate all code related to the given task functions and a Makefile that
    indicates which source files need building and where to look for headers.

    :param tf_spec_jsons_all: List of TaskFunction specification files that
        should be used to generate code
    :param destination: Pre-existing folder to which all code should be written
    :param overwrite: Pre-existing header and source files in destination will
        be overwritten if True
    :param library_path: Full path to Milhoja library installation that will use
        generated code
    :param indent: Number of spaces to use for indent in generated code
    :param makefile_filename: Name with path of makefile to generate.  An
        exception is raised if the file already exists.
    :param logger: Derived from :py:class:`AbcLogger`
    """
    LOG_TAG = "Milhoja Test"

    # ----- ERROR CHECK ARGUMENTS
    dst = Path(destination).resolve()
    makefile = Path(makefile_filename).resolve()
    if not dst.is_dir():
        msg = f"Destination folder {dst} does not exist or is not a folder"
        raise ValueError(msg)
    elif makefile.exists():
        raise ValueError(f"Makefile {makefile} already exists")

    # ----- GENERATE TASK FUNCTION/DATA ITEM PAIRS
    # Keep track of all generated source files
    files_to_compile = []

    logger.log(LOG_TAG, "", LOG_LEVEL_BASIC)
    for tf_spec_json in tf_spec_jsons_all:
        tf_spec = TaskFunction.from_milhoja_json(tf_spec_json)
        logger.log(
            LOG_TAG, f"Generating code for task function {tf_spec.name}",
            LOG_LEVEL_BASIC
        )
        logger.log(LOG_TAG, "-" * 80, LOG_LEVEL_BASIC)

        generate_data_item(
            tf_spec, destination, overwrite, library_path, indent, logger
        )
        generate_task_function(
            tf_spec, destination, overwrite, indent, logger
        )
        logger.log(LOG_TAG, "", LOG_LEVEL_BASIC)

        data_item = tf_spec.data_item
        language = tf_spec.language
        outputs = tf_spec.output_filenames
        to_compile = []
        for _, value in outputs.items():
            # extract check in case source is empty...
            if not value["source"]:
                raise Exception(
                    "Source file provided but file name is null or empty."
                )

            if (language.lower() in ["c++", "fortran"]) and \
                    (data_item.lower() == "datapacket"):
                if not value["source"].startswith("gpu_"):
                    to_compile.append(dst.joinpath(value["source"]))
            else:
                to_compile.append(dst.joinpath(value["source"]))
        files_to_compile += to_compile

    # ----- GENERATE GENERATED-CODE MAKEFILE
    with open(makefile, "w") as fptr:
        fptr.write(f"CXXFLAGS_GENERATED_DEBUG = -I${dst}\n")
        fptr.write(f"CXXFLAGS_GENERATED_PROD  = -I${dst}\n")
        fptr.write("\n")
        fptr.write("SRCS_GENERATED = \\\n")
        for i, filename in enumerate(files_to_compile):
            if i + 1 == len(files_to_compile):
                fptr.write(f"\t{filename}\n")
            else:
                fptr.write(f"\t{filename}, \\\n")

    logger.log(LOG_TAG, f"Generated {makefile}", LOG_LEVEL_BASIC)
    logger.log(LOG_TAG, "", LOG_LEVEL_BASIC)
