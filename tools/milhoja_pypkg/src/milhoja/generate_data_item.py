from . import TileWrapperGenerator_cpp

def generate_data_item(tf_spec, destination, overwrite, verbosity, indent):
    """
    .. todo::
        Add in all other code generators.
    """
    data_item = tf_spec.data_item
    language = tf_spec.language

    if (language.lower() == "c++") and (data_item.lower() == "tilewrapper"):
        generator = TileWrapperGenerator_cpp(tf_spec, verbosity, indent)
        generator.generate_header_code(destination, overwrite)
        generator.generate_source_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()
    else:
        msg = f"Cannot generate data item code for {data_item}/{language}"
        raise ValueError(msg)
