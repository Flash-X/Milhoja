import json

from pathlib import Path

from . import LOG_LEVEL_BASIC
from . import LOG_LEVEL_BASIC_DEBUG
from . import TaskFunction
from . import AbcCodeGenerator
from . import generate_tile_metadata_extraction


class TaskFunctionGenerator_cpu_cpp(AbcCodeGenerator):
    """
    """
    __LOG_TAG = "Milhoja C++/CPU Task Function"

    def __init__(
            self,
            tf_spec,
            log_level,
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
        :param logger: Object for logging code generation details
        :type  logger: CodeGenerationLogger or a class derived from that class
        :param indent: The number of spaces used to define the tab to be used
            in both generated files.
        :type  indent: non-negative int, optional
        """
        outputs = tf_spec.output_filenames
        header_filename = outputs[TaskFunction.CPP_TF_KEY]["header"]
        source_filename = outputs[TaskFunction.CPP_TF_KEY]["source"]

        super().__init__(
            tf_spec,
            header_filename, source_filename,
            TaskFunctionGenerator_cpu_cpp.__LOG_TAG, log_level,
            indent
        )

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
        self.__TILE_DATA_ARRAY_TYPES = ["milhoja::FArray1D", "milhoja::FArray2D",
                                        "milhoja::FArray3D", "milhoja::FArray4D"]
        self.__MIN_DATA_ARRAY_DIM = 1
        self.__MAX_DATA_ARRAY_DIM = len(self.__TILE_DATA_ARRAY_TYPES)

        msg = "Loaded task function specification\n"
        msg += "-" * 80 + "\n"
        msg += str(self)
        self._log(msg, LOG_LEVEL_BASIC_DEBUG)

    @property
    def namespace(self):
        """
        """
        return self._tf_spec.name

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

    def generate_source_code(self, destination, overwrite):
        """
        """
        INDENT = " " * self.indentation

        path = Path(destination).resolve()
        if not path.is_dir():
            raise ValueError(f"{path} is not a folder or does not exist")
        source_filename = path.joinpath(self.source_filename)

        msg = f"Generating C++ Source {source_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        if (not overwrite) and source_filename.exists():
            raise ValueError(f"{source_filename} already exists")

        tile_name = f"Tile_{self._tf_spec.name}"

        json_fname = self.specification_filename

        with open(source_filename, "w") as fptr:
            # ----- HEADER INCLUSION
            # Task function's header file
            fptr.write(f'#include "{self.header_filename}"\n')
            fptr.write(f'#include "Tile_{self.header_filename}"\n')
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
            for subroutine in self._tf_spec.internal_subroutines:
                header = self._tf_spec.subroutine_header(subroutine)
                headers_all = headers_all.union(set([header]))

            for header in sorted(headers_all):
                fptr.write(f'#include "{header}"\n')
            fptr.write("\n")

            # ----- FUNCTION DECLARATION
            fptr.write(f"void  {self.namespace}::taskFunction(const int threadId,\n")
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

            # ----- EXTRACT TASK FUNCTION TILE METADATA FROM TILE
            metadata_all = self._tf_spec.tile_metadata_arguments
            code = generate_tile_metadata_extraction(self._tf_spec, "tileDesc")
            for line in code:
                fptr.write(f"{INDENT}{line}\n")

            # ----- EXTRACT ALL OTHER TASK FUNCTION ARGUMENTS FROM TILE
            for arg in self._tf_spec.argument_list:
                arg_spec = self._tf_spec.argument_specification(arg)
                src = arg_spec["source"].lower()

                if src in metadata_all:
                    pass
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

                    if "extents_in" in arg_spec:
                        extents_in = self.__parse_extents_spec(arg_spec["extents_in"])
                        dimension = len(extents_in)
                    elif "extents_out" in arg_spec:
                        extents_out = self.__parse_extents_spec(arg_spec["extents_out"])
                        dimension = len(extents_out)
                    else:
                        raise ValueError(f"No extents given for grid_data {arg}")
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
                        fptr.write(f"{INDENT*3} static_cast<{arg_type}*>({tile_name}::{arg}_)\n")
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
                        fptr.write(f"{INDENT*3} static_cast<{arg_type}*>({tile_name}::{arg}_)\n")
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
            # We require a flat call graph for CPU task functions
            # TODO: Confirm this
            for subroutine in self._tf_spec.internal_subroutines:
                arg_list = \
                    self._tf_spec.subroutine_actual_arguments(subroutine)
                fptr.write(f"{INDENT}{subroutine}(\n")
                for arg in arg_list[:-1]:
                    fptr.write(f"{INDENT*5}{arg},\n")
                fptr.write(f"{INDENT*5}{arg_list[-1]});\n")

            # ----- CLOSE TASK FUNCTION DEFINITION
            fptr.write("}")

    def generate_header_code(self, destination, overwrite):
        """
        """
        INDENT = " " * self.indentation

        path = Path(destination).resolve()
        if not path.is_dir():
            raise ValueError(f"{path} is not a folder or does not exist")
        header_filename = path.joinpath(self.header_filename)

        msg = f"Generating C++ Header {header_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        if (not overwrite) and header_filename.exists():
            raise ValueError(f"{header_filename} already exists")

        basename = Path(header_filename.name).stem
        hdr_macro = f"MILHOJA_GENERATED_{basename.upper()}_H__"

        with open(header_filename, "w") as fptr:
            fptr.write(f"#ifndef {hdr_macro}\n")
            fptr.write(f"#define {hdr_macro}\n")
            fptr.write("\n")
            fptr.write("#include <Milhoja_DataItem.h>\n")
            fptr.write("\n")

            fptr.write(f"namespace {self.namespace} {{\n")
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
