from pathlib import Path

from . import LOG_LEVEL_BASIC
from . import LOG_LEVEL_BASIC_DEBUG
from . import TaskFunction
from . import AbcCodeGenerator


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

    def __generate_metadata_extraction(self, task_function, tile_desc):
        """
        """
        code = []
    
        args_all = task_function.dummy_arguments
        metadata_all = task_function.tile_metadata_arguments
    
        # ----- ADD TILEMETADATA NEEDED INTERNALLY
        # Some tile metadata can only be accessed using other metadata.
        # Add dependent metadata if not already in list.
        #
        # TODO: Should we have a function that generates variable names since the
        # same MH_INTERNAL_* variables used here are also used in another code
        # generator.
        internal = {}
        for arg in args_all:
            spec = task_function.argument_specification(arg)
    
            dependents = ["tile_coordinates", "tile_faceAreas", "tile_cellVolumes"]
            if spec["source"] in dependents:
                if "tile_level" not in metadata_all:
                    variable = "MH_INTERNAL_level"
                    if variable not in internal:
                        internal[variable] = {"source": "tile_level"}
                        metadata_all["tile_level"] = [variable]
    
                for point in ["lo", "hi"]:
                    key = spec[point].strip().lower()
                    if key not in metadata_all:
                        assert key.startswith("tile_")
                        short = key.replace("tile_", "")
                        assert short in ["lo", "hi", "lbound", "ubound"]
    
                        variable  = f"MH_INTERNAL_{short}"
                        if variable not in internal:
                            internal[variable] = {"source": key}
                            metadata_all[key] = [variable]
    
        # ----- EXTRACT INDEPENDENT METADATA
        # TODO: This is for CPU/C++
        order = [("tile_gridIndex", "const int", "gridIndex"),
                 ("tile_level", "const unsigned int", "level"),
                 ("tile_lo", "const milhoja::IntVect", "lo"),
                 ("tile_hi", "const milhoja::IntVect", "hi"),
                 ("tile_lbound", "const milhoja::IntVect", "loGC"),
                 ("tile_ubound", "const milhoja::IntVect", "hiGC"),
                 ("tile_deltas", "const milhoja::RealVect", "deltas")]
        for key, arg_type, getter in order:
            if key in metadata_all:
                arg = metadata_all[key]
                assert len(arg) == 1
                arg = arg[0]
    
                line = f"{arg_type}   {arg} = {tile_desc}->{getter}();"
                code.append(line)
    
        # ----- CREATE THREAD-PRIVATE INTERNAL SCRATCH
        if "tile_cellVolumes" in metadata_all:
            arg_list = metadata_all["tile_cellVolumes"]
            assert len(arg_list) == 1
            arg = arg_list[0]
            wrapper = f"Tile_{task_function.name}"
            code += [
                f"milhoja::Real*   MH_INTERNAL_cellVolumes_ptr =",
                f"\tstatic_cast<milhoja::Real*>({wrapper}::MH_INTERNAL_cellVolumes_)",
                f"\t+ {wrapper}::MH_INTERNAL_CELLVOLUMES_SIZE_ * threadId;"
            ]
    
        # ----- EXTRACT DEPENDENT METADATA
        axis_mh = {"i": "milhoja::Axis::I",
                   "j": "milhoja::Axis::J",
                   "k": "milhoja::Axis::K"}
        edge_mh = {"center": "milhoja::Edge::Center"}
    
        if "tile_coordinates" in metadata_all:
            arg_list = metadata_all["tile_coordinates"]
            assert len(arg_list) <= 3
            for arg in arg_list:
                spec = task_function.argument_specification(arg)
                axis = axis_mh[spec["axis"].lower()]
                edge = edge_mh[spec["edge"].lower()]
                level = metadata_all["tile_level"][0]
                lo = metadata_all[spec["lo"]][0]
                hi  = metadata_all[spec["hi"]][0]
                code += [
                    f"const milhoja::FArray1D  {arg} =",
                    "\tmilhoja::Grid::instance().getCellCoords(",
                    f"\t\t{axis},",
                    f"\t\t{edge},",
                    f"\t\t{level},",
                    f"\t\t{lo}, {hi});"
                ]
    
        if "tile_faceAreas" in metadata_all:
            raise NotImplementedError("No test case yet for face areas")
    
        if "tile_cellVolumes" in metadata_all:
            arg_list = metadata_all["tile_cellVolumes"]
            assert len(arg_list) == 1
            arg = arg_list[0]
            spec = task_function.argument_specification(arg)
            level = metadata_all["tile_level"][0]
            lo = metadata_all[spec["lo"]][0]
            hi  = metadata_all[spec["hi"]][0]
            code += [
                f"Grid::instance().fillCellVolumes(",
                f"\t{level},",
                f"\t{lo},",
                f"\t{hi},",
                 "\tMH_INTERNAL_cellVolumes_ptr);",
                f"const milhoja::FArray3D  {arg}{{",
                 "\t\tMH_INTERNAL_cellVolumes_ptr,",
                f"\t\t{lo},",
                f"\t\t{hi}}};"
            ]
    
        return code


    def generate_source_code(self, destination, overwrite):
        """
        .. todo::
            * Only load those header files that are really needed
            * Why do external arguments need to specify name?  Why not use arg?
            * Once the Milhoja grid interface is expanded to account
              for multiple MFabs for each data type, improved the grid_data
              section.  Also, we should be able to get any data using
              tileDesc->data(MFab type, ID)
            * Scratch data section is hardcoded.  OK for now since CPU task
              functions are presently just for test suite.  See Issue #59.
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
            headers_all = set()
            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
                    assert len(node) == 1
                    header = \
                        self._tf_spec.subroutine_interface_file(subroutine)
                    headers_all = headers_all.union(set([header]))

            for header in sorted(headers_all):
                fptr.write(f'#include "{header}"\n')
            fptr.write("\n")

            # ----- FUNCTION DECLARATION
            fptr.write(f"void  {self.namespace}::taskFunction(const int threadId,\n")
            fptr.write(f"{INDENT*5}milhoja::DataItem* dataItem) {{\n")

            # ----- USE IN NAMESPACES
            for namespace in ["milhoja"]:
                fptr.write(f"{INDENT}using namespace {namespace};\n")
            fptr.write("\n")

            # ----- ACCESS GIVEN TILE DESCRIPTOR
            fptr.write(f"{INDENT}{tile_name}*  wrapper = dynamic_cast<{tile_name}*>(dataItem);\n")
            fptr.write(f"{INDENT}milhoja::Tile*  tileDesc = wrapper->tile_.get();\n")
            fptr.write("\n")

            # ----- EXTRACT EXTERNAL VARIABLES
            for arg in sorted(self._tf_spec.external_arguments):
                arg_spec = self._tf_spec.argument_specification(arg)
                name = arg_spec["name"]
                param_type = arg_spec["type"]
                extents = arg_spec["extents"]
                if len(extents) == 0:
                    fptr.write(f"{INDENT}{param_type}& {name} = wrapper->{name}_;\n")
                else:
                    raise NotImplementedError("Need arrays")

            # ----- EXTRACT TASK FUNCTION TILE METADATA FROM TILE
            metadata_all = self._tf_spec.tile_metadata_arguments
            code = self.__generate_metadata_extraction(self._tf_spec, "tileDesc")
            for line in code:
                fptr.write(f"{INDENT}{line}\n")

            # ----- EXTRACT GRID DATA POINTERS
            grid_args = self._tf_spec.tile_in_arguments
            grid_args = grid_args.union(self._tf_spec.tile_in_out_arguments)
            grid_args = grid_args.union(self._tf_spec.tile_out_arguments)
            for arg in sorted(grid_args):
                arg_spec = self._tf_spec.argument_specification(arg)
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

#                fptr.write(f'{_INDENT}{arg_type}  {arg} = tileDesc->data("{index_space}", {mfab_idx});\n')
                if index_space == "CENTER":
                    dimension = 4
                    arg_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]
                    fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->data();\n")
                elif index_space == "FLUXX":
                    dimension = 4
                    arg_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]
                    fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->fluxData(milhoja::Axis::I);\n")
                elif index_space == "FLUXY":
                    dimension = 4
                    arg_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]
                    fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->fluxData(milhoja::Axis::J);\n")
                elif index_space == "FLUXZ":
                    dimension = 4
                    arg_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]
                    fptr.write(f"{INDENT}{arg_type}  {arg} = tileDesc->fluxData(milhoja::Axis::K);\n")
                else:
                    raise NotImplementedError("This should never happen")

            # ----- EXTRACT SCRATCH VARIABLES
            for arg in sorted(self._tf_spec.scratch_arguments):
                arg_spec = self._tf_spec.argument_specification(arg)
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

            fptr.write("\n")

            # ----- CALL SUBROUTINES
            # We require a flat call graph for CPU task functions
            # TODO: Confirm this
            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
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
