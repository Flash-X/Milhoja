from . import TaskFunctionGenerator_cpu_cpp


def generate_task_function(tf_spec, destination, overwrite, verbosity, indent):
    """
    .. todo::
        Add in all other code generators.
    """
    processor = tf_spec.processor
    language = tf_spec.language

    if (language.lower() == "c++") and (processor.lower() == "cpu"):
        generator = TaskFunctionGenerator_cpu_cpp(tf_spec, verbosity, indent)
        generator.generate_header_code(destination, overwrite)
        generator.generate_source_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()
    else:
        msg = f"Cannot generate task function code for {processor}/{language}"
        raise ValueError(msg)
