from . import MILHOJA_JSON_FORMAT
from . import TaskFunction
from . import TileWrapperGenerator

def generate_data_item(
        filename, specification_format,
        header_filename, source_filename,
        verbosity, indent
    ):
    """
    TODO: Add in all other code generators.
    """
    if specification_format.lower() == MILHOJA_JSON_FORMAT.lower():
        tf_spec = TaskFunction.from_milhoja_json(filename)
        device = tf_spec.device_information["device"]
        language = tf_spec.language

        if (language.lower() == "c++") and (device.lower() == "cpu"):
            generator = TileWrapperGenerator.from_json(
                            filename,
                            header_filename,
                            source_filename,
                            verbosity,
                            indent
                        )
            generator.generate_header_code()
            generator.generate_source_code()

            assert header_filename.is_file()
            assert source_filename.is_file()
        else:
            msg = f"Cannot generate data item code for {device}/{language}"
            raise ValueError(msg)
    else:
        msg = f"Unknown specification format {specification_format}"
        raise ValueError(msg)
