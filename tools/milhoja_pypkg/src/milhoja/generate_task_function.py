from . import TaskFunctionGenerator_cpu_cpp


def generate_task_function(tf_spec, destination, overwrite, indent, logger):
    """
    .. todo::
        Add in all other code generators.
    """
    processor = tf_spec.processor
    offloading = tf_spec.computation_offloading
    language = tf_spec.language
    data_item = tf_spec.data_item

    if (language.lower() == "c++") and (processor.lower() == "cpu"):
        generator = TaskFunctionGenerator_cpu_cpp(tf_spec, indent, logger)
        generator.generate_header_code(destination, overwrite)
        generator.generate_source_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()
    elif (language.lower() in ["c++", "fortran"]) and \
            (data_item.lower() == "datapacket"):
        logger.warn("Milhoja TF", "No TF generation for use with DataPackets")
    else:
        msg = f"Cannot generate task function code for {processor}/{language}"
        raise ValueError(msg)
