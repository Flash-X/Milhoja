import json

from pathlib import Path

from . import CodeGenerationLogger


class CppTaskFunctionGenerator(object):
    """
    """
    # ----- INSTANTIATION CLASS METHODS
    @classmethod
    def from_json(
            cls,
            tf_spec_json_filename,
            header_filename,
            source_filename,
            logger,
            indent=4
            ):
        """
        Instantiate an object and initialize it with the contents of the given
        JSON-format file, which contains all configuration information needed
        to generate the associated C++ task function.

        See the constructor's documentation for more information.

        :param tf_spec_json_filename: Name of the JSON-format file
        :type  tf_spec_json_filename: str
        :param header_filename: Name of the C++ header file to generate
        :type  header_filename: str
        :param source_filename: Name of the C++ source file to generate
        :type  source_filename: str
        :param logger: Object for logging code generation details
        :type  logger: CodeGenerationLogger or a class derived from that class
        :param indent: The number of spaces used to define the tab to be used
            in both generated files.
        :type  indent: non-negative int, optional
        :return: The generator object ready for use
        :rtype: CppTaskFunctionGenerator
        """
        json_fname = Path(tf_spec_json_filename).resolve()
        if not json_fname.is_file():
            raise ValueError(f"{json_fname} does not exist or is not a file")

        with open(json_fname, "r") as fptr:
            tf_spec = json.load(fptr)
        if "format" not in tf_spec:
            raise ValueError('"format" not provided in JSON specification')

        tf_fmt, tf_version = tf_spec["format"]
        if tf_fmt.lower() != "milhoja_native_json":
            raise ValueError("Unknown JSON format ({tf_fmt} v{tf_version})")
        elif tf_version.lower() != "1.0.0":
            raise ValueError("Invalid Milhoja-native JSON version (v{tf_version})")

        # TODO: Once we have a class that wraps the task function
        # specification, we should instantiate it here with its .to_json
        # classmethod and pass it to the constructor.  tf_spec is presently the
        # standin for that object.
        generator = CppTaskFunctionGenerator(
                        tf_spec,
                        tf_spec_json_filename,
                        header_filename,
                        source_filename,
                        logger,
                        indent
                    )

        msg = f"Created code generator from JSON file {tf_spec_json_filename}"
        logger.log(msg, CodeGenerationLogger.BASIC_DEBUG_LEVEL)

        return generator

    def __init__(
            self,
            tf_spec,
            tf_spec_filename,
            header_filename,
            source_filename,
            logger,
            indent
            ):
        """
        The basename of the header file is adopted as the name of the task
        function.  It is assumed that the task function will receive a data
        item derived from Milhoja_TileWrapper named Tile_<task function name>
        declared in a header file Tile_<task function name>.h.

        :param tf_spec: The XXX task function specification object
        :type  tf_spec: XXX
        :param tf_spec_filename: Name of the task function specification file
        :type  tf_spec_filename: str
        :param header_filename: Name of the C++ header file to generate
        :type  header_filename: str
        :param source_filename: Name of the C++ source file to generate
        :type  source_filename: str
        :param logger: Object for logging code generation details
        :type  logger: CodeGenerationLogger or a class derived from that class
        :param indent: The number of spaces used to define the tab to be used
            in both generated files.
        :type  indent: non-negative int, optional
        """
        super().__init__()

        # ----- STORE ARGUMENTS
        self.__tf_spec = tf_spec

        self.__spec_fname = Path(tf_spec_filename).resolve()
        self.__hdr_fname = Path(header_filename).resolve()
        self.__src_fname = Path(source_filename).resolve()

        self.__indent = indent

        self.__tf_name = Path(self.__hdr_fname.name).stem

        self.__logger = logger

        # ----- SANITY CHECK ARGUMENTS
        # Since there could be no file at instantiation, but a file could
        # appear before calling a generate method, we don't check file
        # existence here.
        if indent < 0:
            raise ValueError(f"Invalid indent ({indent})")

        # ----- CONSTANTS
        # Keys identify the index space of a MFab available through the Milhoja
        # Tile interface (i.e., data, etc.).  For each space, there may be one
        # or more distinct MFabs managed by Grid.  These are indexed with each
        # class by a different set of positive integers.
        #
        # TODO: This is strange.  This information should be encoded in the
        # library.  Seems like a maintenance nightmare to link the contents
        # here to the library that others might be using.  Should some of this
        # information go into the include folder for us to pick out?  What if
        # people want to use this on a machine different from the platform on
        # which they will run?  Should the contents here be specified based on
        # a given library version?
        self.__AVAILABLE_MFABS = {"CENTER": [1],
                                  "FLUXX":  [1], "FLUXY": [1], "FLUXZ": [1]}

        # ----- CODE GENERATION CONSTANTS
        # TODO: Should we actively construct the tile metadata getters as an
        # optimization so that we aren't calling the same getter several times?
        # TODO: As suggested by Wesley, we should probably have a single
        # "tile_coordinates" that requires extra specifications such as axis,
        # edge, and interior/full
        self.__TILE_METADATA_LUT = {"tile_lo":       ("const milhoja::IntVect",  "tileDesc->lo()"),
                                    "tile_hi":       ("const milhoja::IntVect",  "tileDesc->hi()"),
                                    "tile_lbound":   ("const milhoja::IntVect",  "tileDesc->loGC()"),
                                    "tile_ubound":   ("const milhoja::IntVect",  "tileDesc->hiGC()"),
                                    "tile_deltas":   ("const milhoja::RealVect", "tileDesc->deltas()"),
                                    "tile_xCenters": ("const milhoja::FArray1D",
                                                      "milhoja::Grid::instance().getCellCoords(\n" \
                                                    + "\t\tmilhoja::Axis::I,\n" \
                                                    + "\t\tmilhoja::Edge::Center,\n" \
                                                    + "\t\ttileDesc->level(), \n"
                                                    + "\t\ttileDesc->loGC(), \n"
                                                    + "\t\ttileDesc->hiGC()\n"
                                                    + "\t)"),
                                    "tile_yCenters": ("const milhoja::FArray1D",
                                                      "milhoja::Grid::instance().getCellCoords(\n" \
                                                    + "\t\tmilhoja::Axis::J,\n" \
                                                    + "\t\tmilhoja::Edge::Center,\n" \
                                                    + "\t\ttileDesc->level(), \n"
                                                    + "\t\ttileDesc->loGC(), \n"
                                                    + "\t\ttileDesc->hiGC()\n"
                                                    + "\t)"),
                                    "tile_zCenters": ("const milhoja::FArray1D",
                                                      "milhoja::Grid::instance().getCellCoords(\n" \
                                                    + "\t\tmilhoja::Axis::K,\n" \
                                                    + "\t\tmilhoja::Edge::Center,\n" \
                                                    + "\t\ttileDesc->level(), \n"
                                                    + "\t\ttileDesc->loGC(), \n"
                                                    + "\t\ttileDesc->hiGC()\n"
                                                    + "\t)")}

        self.__TILE_DATA_ARRAY_TYPES = ["milhoja::FArray1D", "milhoja::FArray2D",
                                        "milhoja::FArray3D", "milhoja::FArray4D"]
        self.__MIN_DATA_ARRAY_DIM = 1
        self.__MAX_DATA_ARRAY_DIM = len(self.__TILE_DATA_ARRAY_TYPES)

        msg = "Loaded task function specification\n"
        msg += "-" * 80 + "\n"
        msg += str(self)
        self.__logger.log(msg, CodeGenerationLogger.BASIC_DEBUG_LEVEL)

    def __parse_extents_spec(self, spec):
        """
        TODO: This is generic and really should be in a class for accessing a
        task function specification.
        """
        extents = spec.strip()
        assert extents.startswith("(")
        assert extents.endswith(")")
        extents = extents.lstrip("(").rstrip(")")
        return [int(e) for e in extents.split(",")]

    @property
    def specification_filename(self):
        return self.__spec_fname

    @property
    def header_filename(self):
        return self.__hdr_fname

    @property
    def source_filename(self):
        return self.__src_fname

    @property
    def indentation(self):
        return self.__indent

    @property
    def verbosity_level(self):
        return self.__logger.level

    def generate_source_code(self):
        """
        """
        INDENT = " " * self.__indent

        msg = f"Generating C++ Source {self.__src_fname}"
        self.__logger.log(msg, CodeGenerationLogger.BASIC_LOG_LEVEL)

        if self.__src_fname.exists():
            raise ValueError(f"{self.__src_fname} already exists")

        args_all = self.__tf_spec["argument_list"]
        arg_specs_all = self.__tf_spec["argument_specifications"]

        tile_name = f"Tile_{self.__tf_name}"

        json_fname = self.specification_filename

        with open(self.__src_fname, "w") as fptr:
            # ----- HEADER INCLUSION
            # Task function's header file
            fptr.write(f'#include "{self.__hdr_fname.name}"\n')
            fptr.write(f'#include "Tile_{self.__hdr_fname.name}"\n')
            fptr.write("\n")

            # Milhoja header files
            # TODO: Only load those that are really needed.
            fptr.write("#include <Milhoja.h>\n")
            fptr.write("#include <Milhoja_real.h>\n")
            fptr.write("#include <Milhoja_IntVect.h>\n")
            fptr.write("#include <Milhoja_RealVect.h>\n")
            fptr.write("#include <Milhoja_FArray1D.h>\n")
            fptr.write("#include <Milhoja_FArray2D.h>\n")
            fptr.write("#include <Milhoja_FArray3D.h>\n")
            fptr.write("#include <Milhoja_FArray4D.h>\n")
            fptr.write("#include <Milhoja_axis.h>\n")
            fptr.write("#include <Milhoja_edge.h>\n")
            fptr.write("#include <Milhoja_Tile.h>\n")
            fptr.write("#include <Milhoja_Grid.h>\n")
            fptr.write("\n")

            # Application header files for subroutines
            # We require a flat call stack for CPU task functions
            # TODO: Confirm this
            headers_all = set()
            for subroutine in self.__tf_spec["subroutine_call_stack"]:
                subr_spec = self.__tf_spec["subroutine_specifications"][subroutine]
                header = subr_spec["header_file"]
                headers_all = headers_all.union(set([header]))

            for header in sorted(headers_all):
                fptr.write(f'#include "{header}"\n')
            fptr.write("\n")

            # ----- FUNCTION DECLARATION
            fptr.write(f"void  {self.__tf_name}::taskFunction(const int threadId,\n")
            fptr.write(f"{INDENT*5}milhoja::DataItem* dataItem) {{\n")

            # ----- USE IN NAMESPACES
            # TODO: Try to do this so that all types are explicitly given.
            for namespace in ["milhoja"]:
                fptr.write(f"{INDENT}using namespace {namespace};\n")
            fptr.write("\n")

            # ----- ACCESS GIVEN TILE DESCRIPTOR
            fptr.write(f"{INDENT}{tile_name}*  wrapper = dynamic_cast<{tile_name}*>(dataItem);\n")
            fptr.write(f"{INDENT}milhoja::Tile*  tileDesc = wrapper->tile_.get();\n")
            fptr.write("\n")

            # ----- EXTRACT TASK FUNCTION ARGUMENTS FROM TILE
            for arg in args_all:
                arg_spec = arg_specs_all[arg]
                src = arg_spec["source"]

                if src in self.__TILE_METADATA_LUT:
                    arg_type, getter = self.__TILE_METADATA_LUT[arg]
                    fptr.write(f"{INDENT}{arg_type}  {arg} = {getter};\n")
                elif src == "external":
                    name = arg_spec["name"]
                    param_type = arg_spec["type"]
                    extents = arg_spec["extents"]
                    if len(extents) == 0:
                        fptr.write(f"{INDENT}{param_type}& {name} = wrapper->{name}_;\n")
                    else:
                        raise NotImplementedError("Need arrays")
                elif src == "lbound":
                    raise NotImplementedError("lbound arguments are Fortran-specific")
                elif src == "grid_data":
                    arg_mfab = arg_spec["structure_index"]
                    if len(arg_mfab) != 2:
                        error_msg = f"Invalid structure_index for {arg} in {json_fname}"
                        raise ValueError(error_msg)
                    index_space = arg_mfab[0].upper()
                    mfab_idx = arg_mfab[1]
                    if (index_space not in self.__AVAILABLE_MFABS) or \
                       (mfab_idx not in self.__AVAILABLE_MFABS[index_space]):
                        error_msg = f"{arg_mfab} specified for {arg} in {json_fname}"
                        error_msg += "is not a valid grid data structure"
                        raise ValueError(error_msg)

                    extents_in = self.__parse_extents_spec(arg_spec["extents_in"])
                    extents_out = self.__parse_extents_spec(arg_spec["extents_out"])
                    dimension = len(extents_in)
                    if len(extents_out) != dimension:
                        error_msg = f"Inconsistent extents_* for {arg} in {json_fname}"
                        raise ValueError(msg)
                    if (dimension < self.__MIN_DATA_ARRAY_DIM) or \
                       (dimension > self.__MAX_DATA_ARRAY_DIM):
                        error_msg = f"Invalid dimension for {arg} in {json_fname}"
                        raise ValueError(msg)
                    arg_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]

                    # TODO: Once the Milhoja grid interface is expanded to account
                    # for more MFabs
#                    fptr.write(f'{_INDENT}{arg_type}  {arg} = tileDesc->data("{index_space}", {mfab_idx});\n')
                    if index_space == "CENTER":
                        fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->data();\n")
                    elif index_space == "FLUXX":
                        fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->fluxData(milhoja::Axis::I);\n")
                    elif index_space == "FLUXY":
                        fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->fluxData(milhoja::Axis::J);\n")
                    elif index_space == "FLUXZ":
                        fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->fluxData(milhoja::Axis::K);\n")
                    else:
                        raise NotImplementedError("This should never happen")
                elif src == "scratch":
                    extents = self.__parse_extents_spec(arg_spec["extents"])
                    dimension = len(extents)
                    if (dimension < self.__MIN_DATA_ARRAY_DIM) or \
                       (dimension > self.__MAX_DATA_ARRAY_DIM):
                        error_msg = f"Invalid dimension for {arg} in {json_fname}"
                        raise ValueError(msg)
                    arg_type = arg_spec["type"]
                    array_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]

                    # TODO: We should get this from extents and tile_lo
                    if arg == "hydro_op1_auxc":
                        assert dimension == 3
                        fptr.write(f"{INDENT}milhoja::IntVect    lo_{arg} = milhoja::IntVect{{LIST_NDIM(tile_lo.I()-MILHOJA_K1D,\n")
                        fptr.write(f"{INDENT}                                   tile_lo.J()-MILHOJA_K2D,\n")
                        fptr.write(f"{INDENT}                                   tile_lo.K()-MILHOJA_K3D)}};\n")
                        fptr.write(f"{INDENT}milhoja::IntVect    hi_{arg} = milhoja::IntVect{{LIST_NDIM(tile_hi.I()+MILHOJA_K1D,\n")
                        fptr.write(f"{INDENT}                                   tile_hi.J()+MILHOJA_K2D,\n")
                        fptr.write(f"{INDENT}                                   tile_hi.K()+MILHOJA_K3D)}};\n")
                        fptr.write(f"{INDENT}{arg_type}* ptr_{arg} = \n")
                        fptr.write(f"{INDENT*3} static_cast<milhoja::Real*>({tile_name}::{arg}_)\n")
                        fptr.write(f"{INDENT*3}+ {tile_name}::{arg.upper()}_SIZE_ * threadId;\n")
                        fptr.write(f"{INDENT}{array_type}  {arg} = {array_type}{{ptr_{arg},\n")
                        fptr.write(f"{INDENT*3}lo_{arg},\n")
                        fptr.write(f"{INDENT*3}hi_{arg}}};\n")
                    elif arg == "base_op1_scratch":
                        assert dimension == 4
                        fptr.write(f"{INDENT}milhoja::IntVect    lo_{arg} = milhoja::IntVect{{LIST_NDIM(tile_lo.I(),\n")
                        fptr.write(f"{INDENT}                                   tile_lo.J(),\n")
                        fptr.write(f"{INDENT}                                   tile_lo.K())}};\n")
                        fptr.write(f"{INDENT}milhoja::IntVect    hi_{arg} = milhoja::IntVect{{LIST_NDIM(tile_hi.I(),\n")
                        fptr.write(f"{INDENT}                                   tile_hi.J(),\n")
                        fptr.write(f"{INDENT}                                   tile_hi.K())}};\n")
                        fptr.write(f"{INDENT}{arg_type}* ptr_{arg} = \n")
                        fptr.write(f"{INDENT*3} static_cast<milhoja::Real*>({tile_name}::{arg}_)\n")
                        fptr.write(f"{INDENT*3}+ {tile_name}::{arg.upper()}_SIZE_ * threadId;\n")
                        fptr.write(f"{INDENT}{array_type}  {arg} = {array_type}{{ptr_{arg},\n")
                        fptr.write(f"{INDENT*3}lo_{arg},\n")
                        fptr.write(f"{INDENT*3}hi_{arg},\n")
                        fptr.write(f"{INDENT*3}2}};\n")
                    else:
                        raise ValueError(f"Unknown scratch variable {arg}")
                else:
                    raise ValueError(f"Unknown argument type {src}")

            fptr.write("\n")

            # ----- CALL SUBROUTINES
            # We require a flat call stack for CPU task functions
            # TODO: Confirm this
            for subroutine in self.__tf_spec["subroutine_call_stack"]:
                subr_spec = self.__tf_spec["subroutine_specifications"][subroutine]
                arg_list = subr_spec["argument_list"]
                mapping = subr_spec["argument_mapping"]

                fptr.write(f"{INDENT}{subroutine}(\n")
                for arg in arg_list[:-1]:
                    fptr.write(f"{INDENT*5}{mapping[arg]},\n")
                fptr.write(f"{INDENT*5}{mapping[arg_list[-1]]});\n")

            # ----- CLOSE TASK FUNCTION DEFINITION
            fptr.write("}")

    def generate_header_code(self):
        """
        """
        INDENT = " " * self.__indent

        msg = f"Generating C++ Header {self.__hdr_fname}"
        self.__logger.log(msg, CodeGenerationLogger.BASIC_LOG_LEVEL)

        if self.__hdr_fname.exists():
            raise ValueError(f"{self.__hdr_fname} already exists")

        basename = Path(self.__hdr_fname.name).stem
        hdr_macro = f"CGKIT_GENERATED_{basename.upper()}_H__"

        with open(self.__hdr_fname, "w") as fptr:
            fptr.write(f"#ifndef {hdr_macro}\n")
            fptr.write(f"#define {hdr_macro}\n")
            fptr.write("\n")
            fptr.write("#include <Milhoja_DataItem.h>\n")
            fptr.write("\n")

            fptr.write(f"namespace {self.__tf_name} {{\n")
            fptr.write(f"{INDENT}void  taskFunction(const int threadId,\n")
            fptr.write(f"{INDENT}                   milhoja::DataItem* dataItem);\n")
            fptr.write("};\n")
            fptr.write("\n")

            fptr.write("#endif\n")

    def __str__(self):
        msg = f"Task Function Specification File\t{self.specification_filename}\n"
        msg += f"TileWrapper C++ Header File\t\t{self.header_filename}\n"
        msg += f"TileWrapper C++ Source File\t\t{self.source_filename}\n"
        msg += f"Indentation length\t\t\t{self.indentation}\n"
        msg += f"Verbosity level\t\t\t\t{self.verbosity_level}"

        return msg
