from pathlib import Path

from milhoja import LOG_LEVEL_BASIC
from milhoja import TaskFunction
from milhoja import generate_data_item
from milhoja import generate_task_function


def generate_runtime_cpu_code(tf_specs_all, destination, overwrite,
                              indent, makefile_filename, logger):
    """
    Generate all Runtime/CPU test code derived from the given task functions
    and a Makefile that indicates which source files need building and where to
    look for headers.

    .. todo::
        * Is this really tied to the CPU?  If so, can we make it generic?

    :param tf_specs_all: List of JSON files that specify task functions and
        their data items as needed for test
    :param destination: Pre-existing folder to which all code should be written
    :param overwrite: Pre-existing header and source files in destination will
        be overwritten if True
    :param indent: Number of spaces to use for indent in generated code
    :param makefile_filename: Name with path of makefile to generate.  An
        exception is raised if the file already exists.
    :param logger: Logger derived from milhoja.AbcLogger
    """
    LOG_TAG = "Milhoja Runtime Test"

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
    for tf_spec_fname in tf_specs_all:
        logger.log(
            LOG_TAG, f"Generating code for task function {tf_spec_fname.name}",
            LOG_LEVEL_BASIC
        )
        logger.log(LOG_TAG, "-" * 80, LOG_LEVEL_BASIC)

        tf_spec = TaskFunction.from_milhoja_json(tf_spec_fname)
        generate_data_item(
            tf_spec, destination, overwrite, indent, logger
        )
        generate_task_function(
            tf_spec, destination, overwrite, indent, logger
        )
        logger.log(LOG_TAG, "", LOG_LEVEL_BASIC)

        outputs = tf_spec.output_filenames
        files_to_compile += \
            [dst.joinpath(value["source"]) for _, value in outputs.items()]

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
