import json

from numbers import Integral
from pathlib import Path

from .constants import LOG_LEVEL_BASIC_DEBUG
from .TileWrapperGenerator_cpp import TileWrapperGenerator_cpp
from .DataPacketGenerator import DataPacketGenerator


def generate_data_item(tf_spec, destination, overwrite, library_path,
                       indent, logger):
    """
    Generate all data item code related to the given task function.

    .. todo::
        Add in all other code generators.

    :param tf_spec: TaskFunction specification including data item's
        specification
    :param destination: Pre-existing folder to which all code should be written
    :param overwrite: Pre-existing header and source files in destination will
        be overwritten if True
    :param library_path: Full path to Milhoja library installation that will use
        generated code
    :param indent: Number of spaces to use for indent in generated code
    :param logger: Derived from :py:class:`AbcLogger`
    """
    LOG_TAG = "Milhoja Tools"

    data_item = tf_spec.data_item
    language = tf_spec.language

    if (language.lower() == "c++" or language.lower() == "fortran") and \
            (data_item.lower() == "tilewrapper"):

        generator = TileWrapperGenerator_cpp(tf_spec, indent, logger)
        generator.generate_header_code(destination, overwrite)
        generator.generate_source_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()
    elif (language.lower() == "c++" or language.lower() == "fortran") and \
            (data_item.lower() == "datapacket"):

        library = Path(library_path).resolve()
        sizes_json = library.joinpath("include", "sizes.json")
        if not library.is_dir():
            msg = f"{library_path} does not exist or is not a directory"
            raise ValueError(msg)
        elif not sizes_json.is_file():
            msg = f"{sizes_json} not installed properly in library installation"
            raise RuntimeError(msg)

        msg = f"Loading platform-specific sizes from {sizes_json}"
        logger.log(LOG_TAG, msg, LOG_LEVEL_BASIC_DEBUG)
        with open(sizes_json, "r") as fptr:
            sizes = json.load(fptr)
        expected = {
            "real", "int", "unsigned int", "std::size_t",
            "IntVect", "RealVect",
            "FArray1D", "FArray2D", "FArray3D", "FArray4D",
            "byte_align"
        }
        assert set(sizes) == expected
        for name, sz in sizes.items():
            assert isinstance(sz, Integral)
            assert sz > 0
            msg = f"\tSize of {name} = {sz} bytes"
            logger.log(LOG_TAG, msg, LOG_LEVEL_BASIC_DEBUG)

        generator = DataPacketGenerator(tf_spec, indent, logger, sizes)
        generator.generate_templates(destination, overwrite)
        generator.generate_source_code(destination, overwrite)
        generator.generate_header_code(destination, overwrite)

        assert destination.joinpath(generator.header_filename).is_file()
        assert destination.joinpath(generator.source_filename).is_file()
    else:
        msg = f"Cannot generate data item code for {data_item}/{language}"
        raise ValueError(msg)
