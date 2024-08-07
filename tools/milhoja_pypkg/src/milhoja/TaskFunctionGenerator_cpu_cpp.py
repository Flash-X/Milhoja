from pathlib import Path

from .constants import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    TILE_GRID_INDEX_ARGUMENT,
    TILE_LEVEL_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_DELTAS_ARGUMENT,
    TILE_COORDINATES_ARGUMENT,
    TILE_FACE_AREAS_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT,
    THREAD_INDEX_VAR_NAME,
)
from .TaskFunction import TaskFunction
from .AbcLogger import AbcLogger
from .AbcCodeGenerator import AbcCodeGenerator


class TaskFunctionGenerator_cpu_cpp(AbcCodeGenerator):
    """
    A class for generating final, compilable C++ header and source code for the
    task function specified by the TaskFunction object given at instantiation.
    """
    __LOG_TAG = "Milhoja C++/CPU Task Function"

    def __init__(self, tf_spec, indent, logger):
        """
        Construct an object for use with the task function specified by the
        given specification object.

        :param tf_spec: Specification object derived from TaskFunction
        :param log_level: Milhoja level to use for logging generation
        :param logger: Logger derived from :py:class`milhoja.AbcLogger`
        """
        if not isinstance(tf_spec, TaskFunction):
            raise TypeError("Given tf_spec not derived from TaskFunction")
        elif not isinstance(logger, AbcLogger):
            raise TypeError("Invalid logger type")

        outputs = tf_spec.output_filenames
        header_filename = outputs[TaskFunction.CPP_TF_KEY]["header"]
        source_filename = outputs[TaskFunction.CPP_TF_KEY]["source"]

        super().__init__(
            tf_spec, header_filename, source_filename, indent,
            TaskFunctionGenerator_cpu_cpp.__LOG_TAG, logger
        )

        # ----- CODE GENERATION CONSTANTS
        self.__TILE_DATA_ARRAY_TYPES = [
            "milhoja::FArray1D", "milhoja::FArray2D",
            "milhoja::FArray3D", "milhoja::FArray4D"
        ]

        msgs_all = [
            "Loaded task function specification",
            "-" * 80,
            f"Specification File\t\t{self.specification_filename}",
            f"Task Function Header\t\t{self.header_filename}",
            f"Task Function Source\t\t{self.source_filename}",
            f"Indentation length\t\t{self.indentation}",
            f"Verbosity level\t\t\t{self.verbosity_level}"
        ]
        for msg in msgs_all:
            self._log(msg, LOG_LEVEL_BASIC_DEBUG)

    @property
    def namespace(self):
        """
        """
        return self._tf_spec.name

    def __parse_extents_spec(self, spec):
        """
        .. todo::
            * This is generic and really should be in a class for accessing a
              task function specification.  Make an ArrayInfo class that
              TaskFunction returns for extents instead of string?  Should that
              class also manage lbound?
        """
        extents = spec.strip()
        assert extents.startswith("(")
        assert extents.endswith(")")
        extents = extents.lstrip("(").rstrip(")")
        return [int(e) for e in extents.split(",")]

    def __generate_metadata_extraction(self, task_function, tile_desc):
        """
        .. todo::
            * Should we have a function that generates variable names since the
            same MH_INTERNAL_* variables used here are also used in another
            code generator.
        """
        code = []

        metadata_all = task_function.tile_metadata_arguments

        # ----- ADD TILEMETADATA NEEDED INTERNALLY
        # Some tile metadata can only be accessed using other metadata.
        # Add dependent metadata if not already in list.
        internal = {}
        for arg in task_function.dummy_arguments:
            spec = task_function.argument_specification(arg)

            dependents = [
                TILE_COORDINATES_ARGUMENT,
                TILE_FACE_AREAS_ARGUMENT,
                TILE_CELL_VOLUMES_ARGUMENT
            ]
            if spec["source"] in dependents:
                if TILE_LEVEL_ARGUMENT not in metadata_all:
                    variable = "MH_INTERNAL_level"
                    if variable not in internal:
                        internal[variable] = {"source": TILE_LEVEL_ARGUMENT}
                        metadata_all[TILE_LEVEL_ARGUMENT] = [variable]

                for point in ["lo", "hi"]:
                    key = spec[point].strip()
                    if key not in metadata_all:
                        assert key.startswith("tile_")
                        short = key.replace("tile_", "")
                        assert short in ["lo", "hi", "lbound", "ubound"]

                        variable = f"MH_INTERNAL_{short}"
                        if variable not in internal:
                            internal[variable] = {"source": key}
                            metadata_all[key] = [variable]

        # ----- EXTRACT INDEPENDENT METADATA
        order = [(TILE_GRID_INDEX_ARGUMENT, "const int", "gridIndex"),
                 (TILE_LEVEL_ARGUMENT, "const unsigned int", "level"),
                 (TILE_LO_ARGUMENT, "const milhoja::IntVect", "lo"),
                 (TILE_HI_ARGUMENT, "const milhoja::IntVect", "hi"),
                 (TILE_LBOUND_ARGUMENT, "const milhoja::IntVect", "loGC"),
                 (TILE_UBOUND_ARGUMENT, "const milhoja::IntVect", "hiGC"),
                 (TILE_DELTAS_ARGUMENT, "const milhoja::RealVect", "deltas")]
        for key, arg_type, getter in order:
            if key in metadata_all:
                arg = metadata_all[key]
                assert len(arg) == 1
                arg = arg[0]

                line = f"{arg_type}   {arg} = {tile_desc}->{getter}();"
                code.append(line)

        # ----- CREATE THREAD-PRIVATE INTERNAL SCRATCH
        if TILE_CELL_VOLUMES_ARGUMENT in metadata_all:
            arg_list = metadata_all[TILE_CELL_VOLUMES_ARGUMENT]
            assert len(arg_list) == 1
            arg = arg_list[0]
            wrapper = task_function.data_item_class_name
            code += [
                "milhoja::Real*   MH_INTERNAL_cellVolumes_ptr =",
                f"\tstatic_cast<milhoja::Real*>({wrapper}::MH_INTERNAL_cellVolumes_)",
                f"\t+ {wrapper}::MH_INTERNAL_CELLVOLUMES_SIZE_ * {THREAD_INDEX_VAR_NAME};"
            ]

        # ----- EXTRACT DEPENDENT METADATA
        axis_mh = {"i": "milhoja::Axis::I",
                   "j": "milhoja::Axis::J",
                   "k": "milhoja::Axis::K"}
        edge_mh = {"center": "milhoja::Edge::Center"}

        if TILE_COORDINATES_ARGUMENT in metadata_all:
            arg_list = metadata_all[TILE_COORDINATES_ARGUMENT]
            assert len(arg_list) <= 3
            for arg in arg_list:
                spec = task_function.argument_specification(arg)
                axis = axis_mh[spec["axis"].lower()]
                edge = edge_mh[spec["edge"].lower()]
                level = metadata_all[TILE_LEVEL_ARGUMENT][0]
                lo = metadata_all[spec["lo"]][0]
                hi = metadata_all[spec["hi"]][0]
                code += [
                    f"const milhoja::FArray1D  {arg} =",
                    "\tmilhoja::Grid::instance().getCellCoords(",
                    f"\t\t{axis},",
                    f"\t\t{edge},",
                    f"\t\t{level},",
                    f"\t\t{lo}, {hi});"
                ]

        if TILE_FACE_AREAS_ARGUMENT in metadata_all:
            raise NotImplementedError("No test case yet for face areas")

        if TILE_CELL_VOLUMES_ARGUMENT in metadata_all:
            arg_list = metadata_all[TILE_CELL_VOLUMES_ARGUMENT]
            assert len(arg_list) == 1
            arg = arg_list[0]
            spec = task_function.argument_specification(arg)
            level = metadata_all[TILE_LEVEL_ARGUMENT][0]
            lo = metadata_all[spec["lo"]][0]
            hi = metadata_all[spec["hi"]][0]
            code += [
                "milhoja::Grid::instance().fillCellVolumes(",
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
            * Write comments at top to indicate that this is generated code and
              to provide some traceability of how it was generated.
            * This implementation presently assumes that if you are running on
              the CPU, then your data item must be a TileWrapper.  This will
              not necessarily be true if, for instance, Grid data structures
              are based in remote memory.
            * Only load those header files that are really needed
            * Why do external arguments need to specify name?  Why not use arg?
            * Once the Milhoja grid interface is expanded to account
              for multiple MFabs for each data type, improved the grid_data
              section.
            * Scratch data section is hardcoded.  OK for now since CPU task
              functions are presently just for test suite.  See Issue #59.
            * Since JSON files can map the TF's threadId onto a subroutine
              dummy argument, rename that variable to something that reads more
              clearly in the JSON.
        """
        INDENT = " " * self.indentation

        # Name of tileDesc variable in source.
        # All code below should use TILE_DESC instead of their own string.
        TILE_DESC = "tileDesc"

        dst = Path(destination).resolve()
        if not dst.is_dir():
            raise RuntimeError(f"{dst} is not a folder or does not exist")
        source_filename = dst.joinpath(self.source_filename)

        msg = f"Generating C++ Source {source_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        if (not overwrite) and source_filename.exists():
            raise RuntimeError(f"{source_filename} already exists")

        wrapper = self._tf_spec.data_item_class_name

        with open(source_filename, "w") as fptr:
            # ----- HEADER INCLUSION
            # Task function's header file
            outputs = self._tf_spec.output_filenames
            data_item_header = outputs[TaskFunction.DATA_ITEM_KEY]["header"]
            fptr.write(f'#include "{self.header_filename}"\n')
            fptr.write(f'#include "{data_item_header}"\n')
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
            headers_all = set()
            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
                    header = \
                        self._tf_spec.subroutine_interface_file(subroutine)
                    headers_all = headers_all.union(set([header]))

            for header in sorted(headers_all):
                fptr.write(f'#include "{header}"\n')
            fptr.write("\n")

            # ----- FUNCTION DECLARATION
            fptr.write(f"void  {self.namespace}::taskFunction(const int {THREAD_INDEX_VAR_NAME},\n")
            fptr.write(f"{INDENT*5}milhoja::DataItem* dataItem) {{\n")

            # ----- ACCESS GIVEN TILE DESCRIPTOR
            fptr.write(f"{INDENT}{wrapper}*  wrapper = dynamic_cast<{wrapper}*>(dataItem);\n")
            fptr.write(f"{INDENT}milhoja::Tile*  {TILE_DESC} = wrapper->tile_.get();\n")
            fptr.write("\n")

            # ----- EXTRACT EXTERNAL VARIABLES
            for arg in sorted(self._tf_spec.external_arguments):
                arg_spec = self._tf_spec.argument_specification(arg)
                arg_type = arg_spec["type"]
                extents = arg_spec["extents"]
                if extents == "()":
                    fptr.write(f"{INDENT}const {arg_type}& {arg} = wrapper->{arg}_;\n")
                else:
                    raise NotImplementedError("Need arrays")

            # ----- EXTRACT TASK FUNCTION TILE METADATA FROM TILE
            code = self.__generate_metadata_extraction(
                        self._tf_spec, TILE_DESC
                   )
            for line in code:
                fptr.write(f"{INDENT}{line}\n")

            # ----- EXTRACT GRID DATA POINTERS
            tile_in = self._tf_spec.tile_in_arguments
            grid_args = tile_in.union(self._tf_spec.tile_in_out_arguments)
            grid_args = grid_args.union(self._tf_spec.tile_out_arguments)
            for arg in sorted(grid_args):
                arg_spec = self._tf_spec.argument_specification(arg)
                arg_mfab = arg_spec["structure_index"]
                assert len(arg_mfab) == 2
                index_space, mfab_idx = arg_mfab
                assert mfab_idx == 1

                dimension = 4
                arg_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]
                if arg in tile_in:
                    arg_type = f"const {arg_type}"

                if index_space.upper() == "CENTER":
                    fptr.write(f"{INDENT}{arg_type}  {arg} = {TILE_DESC}->data();\n")
                elif index_space.upper() == "FLUXX":
                    fptr.write(f"{INDENT}{arg_type}  {arg} = {TILE_DESC}->fluxData(milhoja::Axis::I);\n")
                elif index_space.upper() == "FLUXY":
                    fptr.write(f"{INDENT}{arg_type}  {arg} = {TILE_DESC}->fluxData(milhoja::Axis::J);\n")
                elif index_space.upper() == "FLUXZ":
                    fptr.write(f"{INDENT}{arg_type}  {arg} = {TILE_DESC}->fluxData(milhoja::Axis::K);\n")
                else:
                    raise NotImplementedError("This should never happen")

            # ----- EXTRACT SCRATCH VARIABLES
            for arg in sorted(self._tf_spec.scratch_arguments):
                arg_spec = self._tf_spec.argument_specification(arg)
                extents = self.__parse_extents_spec(arg_spec["extents"])
                dimension = len(extents)
                array_type = self.__TILE_DATA_ARRAY_TYPES[dimension - 1]
                arg_type = arg_spec["type"]

                if arg == "scratch_hydro_op1_auxC":
                    assert dimension == 3
                    fptr.write(f"{INDENT}const milhoja::IntVect    lo_{arg} = milhoja::IntVect{{LIST_NDIM(tile_lo.I()-MILHOJA_K1D,\n")
                    fptr.write(f"{INDENT}                                   tile_lo.J()-MILHOJA_K2D,\n")
                    fptr.write(f"{INDENT}                                   tile_lo.K()-MILHOJA_K3D)}};\n")
                    fptr.write(f"{INDENT}const milhoja::IntVect    hi_{arg} = milhoja::IntVect{{LIST_NDIM(tile_hi.I()+MILHOJA_K1D,\n")
                    fptr.write(f"{INDENT}                                   tile_hi.J()+MILHOJA_K2D,\n")
                    fptr.write(f"{INDENT}                                   tile_hi.K()+MILHOJA_K3D)}};\n")
                    fptr.write(f"{INDENT}{arg_type}* ptr_{arg} = \n")
                    fptr.write(f"{INDENT*3} static_cast<{arg_type}*>({wrapper}::{arg}_)\n")
                    fptr.write(f"{INDENT*3}+ {wrapper}::{arg.upper()}_SIZE_ * {THREAD_INDEX_VAR_NAME};\n")
                    fptr.write(f"{INDENT}{array_type}  {arg} = {array_type}{{ptr_{arg},\n")
                    fptr.write(f"{INDENT*3}lo_{arg},\n")
                    fptr.write(f"{INDENT*3}hi_{arg}}};\n")
                elif arg == "scratch_base_op1_scratch3D":
                    fptr.write(f"{INDENT}const milhoja::IntVect    lo_{arg} = milhoja::IntVect{{LIST_NDIM(tile_lo.I(),\n")
                    fptr.write(f"{INDENT}                                   tile_lo.J(),\n")
                    fptr.write(f"{INDENT}                                   tile_lo.K())}};\n")
                    fptr.write(f"{INDENT}const milhoja::IntVect    hi_{arg} = milhoja::IntVect{{LIST_NDIM(tile_hi.I(),\n")
                    fptr.write(f"{INDENT}                                   tile_hi.J(),\n")
                    fptr.write(f"{INDENT}                                   tile_hi.K())}};\n")
                    fptr.write(f"{INDENT}{arg_type}* ptr_{arg} = \n")
                    fptr.write(f"{INDENT*3} static_cast<{arg_type}*>({wrapper}::{arg}_)\n")
                    fptr.write(f"{INDENT*3}+ {wrapper}::{arg.upper()}_SIZE_ * {THREAD_INDEX_VAR_NAME};\n")
                    fptr.write(f"{INDENT}{array_type}  {arg} = {array_type}{{ptr_{arg},\n")
                    fptr.write(f"{INDENT*3}lo_{arg},\n")
                    fptr.write(f"{INDENT*3}hi_{arg}}};\n")
                elif arg == "scratch_base_op1_scratch4D":
                    fptr.write(f"{INDENT}const milhoja::IntVect    lo_{arg} = milhoja::IntVect{{LIST_NDIM(tile_lo.I(),\n")
                    fptr.write(f"{INDENT}                                   tile_lo.J(),\n")
                    fptr.write(f"{INDENT}                                   tile_lo.K())}};\n")
                    fptr.write(f"{INDENT}const milhoja::IntVect    hi_{arg} = milhoja::IntVect{{LIST_NDIM(tile_hi.I(),\n")
                    fptr.write(f"{INDENT}                                   tile_hi.J(),\n")
                    fptr.write(f"{INDENT}                                   tile_hi.K())}};\n")
                    fptr.write(f"{INDENT}{arg_type}* ptr_{arg} = \n")
                    fptr.write(f"{INDENT*3} static_cast<{arg_type}*>({wrapper}::{arg}_)\n")
                    fptr.write(f"{INDENT*3}+ {wrapper}::{arg.upper()}_SIZE_ * {THREAD_INDEX_VAR_NAME};\n")
                    fptr.write(f"{INDENT}{array_type}  {arg} = {array_type}{{ptr_{arg},\n")
                    fptr.write(f"{INDENT*3}lo_{arg},\n")
                    fptr.write(f"{INDENT*3}hi_{arg},\n")
                    fptr.write(f"{INDENT*3}2}};\n")
                else:
                    raise ValueError(f"Unknown scratch variable {arg}")

            fptr.write("\n")

            # ----- CALL SUBROUTINES
            # We require a flat call graph for CPU task functions
            for node in self._tf_spec.internal_subroutine_graph:
                assert len(node) == 1
                for subroutine in node:
                    arg_list = \
                        self._tf_spec.subroutine_actual_arguments(subroutine)
                    fptr.write(f"{INDENT}{subroutine}(\n")
                    for arg in arg_list[:-1]:
                        fptr.write(f"{INDENT*5}{arg},\n")
                    fptr.write(f"{INDENT*5}{arg_list[-1]});\n")

            # ----- CLOSE TASK FUNCTION DEFINITION
            fptr.write("}\n")

    def generate_header_code(self, destination, overwrite):
        """
        .. todo::
            * Write comments at top to indicate that this is generated code and
              to provide some traceability of how it was generated.
        """
        INDENT = " " * self.indentation

        dst = Path(destination).resolve()
        if not dst.is_dir():
            raise RuntimeError(f"{dst} is not a folder or does not exist")
        header_filename = dst.joinpath(self.header_filename)
        hdr_macro = f"MILHOJA_GENERATED_{header_filename.stem.upper()}_H__"

        msg = f"Generating C++ Header {header_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        if (not overwrite) and header_filename.exists():
            raise RuntimeError(f"{header_filename} already exists")

        with open(header_filename, "w") as fptr:
            fptr.write(f"#ifndef {hdr_macro}\n")
            fptr.write(f"#define {hdr_macro}\n")
            fptr.write("\n")
            fptr.write("#include <Milhoja_DataItem.h>\n")
            fptr.write("\n")

            fptr.write(f"namespace {self.namespace} {{\n")
            fptr.write(f"{INDENT}void  taskFunction(const int {THREAD_INDEX_VAR_NAME},\n")
            fptr.write(f"{INDENT}                   milhoja::DataItem* dataItem);\n")
            fptr.write("};\n")
            fptr.write("\n")

            fptr.write("#endif\n")
