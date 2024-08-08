from . import TaskFunctionGenerator_cpu_cpp
from . import TaskFunctionGenerator_cpu_F
from . import TaskFunctionGenerator_OpenACC_F
from . import TaskFunctionC2FGenerator_cpu_F
from . import TaskFunctionCpp2CGenerator_cpu_F
from . import TileWrapperModGenerator
from . import TaskFunctionC2FGenerator_OpenACC_F
from . import TaskFunctionCpp2CGenerator_OpenACC_F
from . import DataPacketModGenerator


def generate_task_function(tf_spec, destination, overwrite, indent, logger):
    """
    Generates Task Function code based on the given input parameters.

    :param tf_spec: The Task Function Specification object
    :param destination: The pre-existing destination folder for the files
    :param overwrite: Flag for overwriting any files at the destination if set
    :param indent: The number of spaces to set as the indent
    :param logger: The logging object derived from :py:class:`AbcLogger`
    """
    processor = tf_spec.processor.lower()
    offloading = tf_spec.computation_offloading.lower()
    language = tf_spec.language
    data_item = tf_spec.data_item.lower()

    # todo:: This should use a mapping to call a function to determine the
    #        correct generator combo for cleaner code.
    if (language.lower() == "c++") and (processor == "cpu"):
        generator = TaskFunctionGenerator_cpu_cpp(tf_spec, indent, logger)
        generator.generate_header_code(destination, overwrite)
        generator.generate_source_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()

    elif (language.lower() == "fortran") and (processor == "cpu"):
        generator = TaskFunctionGenerator_cpu_F(tf_spec, indent, logger)
        generator.generate_source_code(destination, overwrite)
        assert destination.joinpath(generator.source_filename).is_file()

        cpp2c_generator = \
            TaskFunctionCpp2CGenerator_cpu_F(tf_spec, indent, logger)
        cpp2c_generator.generate_source_code(destination, overwrite)
        assert destination.joinpath(generator.source_filename).is_file()

        c2f_generator = \
            TaskFunctionC2FGenerator_cpu_F(tf_spec, indent, logger)
        c2f_generator.generate_source_code(destination, overwrite)
        assert destination.joinpath(generator.source_filename).is_file()

        mod_generator = TileWrapperModGenerator(tf_spec, indent, logger)
        mod_generator.generate_source_code(destination, overwrite)
        assert destination.joinpath(mod_generator.source_filename).is_file()

    elif (language.lower() == "fortran") and (offloading == "openacc"):
        generator = TaskFunctionGenerator_OpenACC_F(tf_spec, indent, logger)
        generator.generate_source_code(destination, overwrite)
        assert destination.joinpath(generator.source_filename).is_file()

        generator = \
            TaskFunctionC2FGenerator_OpenACC_F(tf_spec, indent, logger)
        generator.generate_source_code(destination, overwrite)
        assert destination.joinpath(generator.source_filename).is_file()

        generator = \
            TaskFunctionCpp2CGenerator_OpenACC_F(tf_spec, indent, logger)
        generator.generate_source_code(destination, overwrite)
        assert destination.joinpath(generator.source_filename).is_file()

        generator = DataPacketModGenerator(tf_spec, indent, logger)
        generator.generate_source_code(destination, indent)
        assert destination.joinpath(generator.source_filename).is_file()

    elif (language.lower() in ["c++", "fortran"]) and \
            (data_item == "datapacket"):
        logger.warn("Milhoja TF", "No TF generation for use with DataPackets")

    else:
        msg = f"Cannot generate task function code for {processor}/{language}"
        raise ValueError(msg)
