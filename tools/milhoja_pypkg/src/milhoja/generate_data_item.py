from . import TileWrapperGenerator_cpp
from . import DataPacketGenerator


def generate_data_item(tf_spec, destination, overwrite, indent, logger):
    """
    .. todo::
        Add in all other code generators.
    """
    data_item = tf_spec.data_item
    language = tf_spec.language

    if (language.lower() == "c++") and (data_item.lower() == "tilewrapper"):
        generator = TileWrapperGenerator_cpp(tf_spec, indent, logger)
        generator.generate_header_code(destination, overwrite)
        generator.generate_source_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()
    elif (language.lower() == "c++" or language.lower() == "fortran") and \
            (data_item.lower() == "datapacket"):

        # ..todo::
        #   * Pass in sizes dictionary to data packet generator.
        generator = DataPacketGenerator(
           tf_spec, indent, logger, {}
        )
        generator.generate_templates(destination, overwrite)
        generator.generate_source_code(destination, overwrite)
        generator.generate_header_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()
    else:
        msg = f"Cannot generate data item code for {data_item}/{language}"
        raise ValueError(msg)
