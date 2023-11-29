from pathlib import Path

from . import TaskFunction
from . import AbcCodeGenerator
from . import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG,
    TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_CELL_VOLUMES_ARGUMENT
)


class TileWrapperGenerator_cpp(AbcCodeGenerator):
    """
    A class for generating final, compilable C++ header and source code that
    defines a Milhoja_TileWrapper derived class for constructing the data item
    passed to the task function specified by the specification object given at
    instantiation.
    """
    __LOG_TAG = "Milhoja C++ Tile Wrapper"

    def __init__(
            self,
            tf_spec,
            indent,
            logger
            ):
        """
        Construct an object for use with the task function specified by the
        given specification object.

        :param tf_spec: Specification object derived from TaskFunction
        :param indent: Number of spaces in tab to be used in generated files
        :param logger: Concrete logger derived from AbcLogger
        """
        if not isinstance(tf_spec, TaskFunction):
            raise TypeError("Given tf_spec not derived from TaskFunction")

        outputs = tf_spec.output_filenames
        header_filename = outputs[TaskFunction.DATA_ITEM_KEY]["header"]
        source_filename = outputs[TaskFunction.DATA_ITEM_KEY]["source"]

        super().__init__(
            tf_spec, header_filename, source_filename, indent,
            TileWrapperGenerator_cpp.__LOG_TAG, logger
        )

        # ----- DETERMINE INTERNAL SCRATCH NEEDED & STORE
        self.__internal_scratch_specs = {}
        for arg in self._tf_spec.dummy_arguments:
            arg_spec = self._tf_spec.argument_specification(arg)
            if arg_spec["source"] == TILE_CELL_VOLUMES_ARGUMENT:
                nguard = self._tf_spec.n_guardcells
                extents = list(self._tf_spec.block_interior_shape)
                for i in range(self._tf_spec.grid_dimension):
                    if arg_spec["lo"] == TILE_LBOUND_ARGUMENT:
                        extents[i] += nguard
                    if arg_spec["hi"] == TILE_UBOUND_ARGUMENT:
                        extents[i] += nguard
                extents = "(" + \
                          ", ".join([str(each) for each in extents]) + \
                          ")"

                name = "MH_INTERNAL_cellVolumes"
                assert name not in self.__internal_scratch_specs
                self.__internal_scratch_specs[name] = {
                    "type": "milhoja::Real",
                    "extents": extents
                }

        # ----- SANITY CHECK ARGUMENTS
        # Since there could be no file at instantiation, but a file could
        # appear before calling a generate method, we don't check file
        # existence here.

        msg = "Loaded task function specification\n"
        msg += "-" * 80 + "\n"
        msg += str(self)
        self._log(msg, LOG_LEVEL_BASIC_DEBUG)

    @property
    def class_name(self):
        """
        """
        return self._tf_spec.data_item_class_name

    @property
    def __scratch_variables(self):
        """
        Code in this class should likely use this instead of accessing scratch
        argument directly from the TaskFunction object.

        :return: Set of task function's scratch arguments and all
            Milhoja-internal scratch variables needed
        """
        tf_arg = self._tf_spec.scratch_arguments
        internal = self.__internal_scratch_specs.keys()
        return tf_arg.union(internal)

    def __scratch_specification(self, arg):
        """
        Code in this class should likely use this instead of accessing scratch
        argument specifications directly from the TaskFunction object.

        :return: Specification for the given task function scratch argument or
            Milhoja-internal scratch variable.
        """
        if arg in self.__internal_scratch_specs:
            return self.__internal_scratch_specs[arg]

        return self._tf_spec.argument_specification(arg)

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

    def __generate_constructor_declaration(self):
        """
        Constructor argument declaration needed identically for both the header
        and source generation.

        :return: Fully-formatted string ready for immediate insertion in
            constructor declaration/definition.
        """
        INDENT = " " * self.indentation

        constructor_args = self._tf_spec.constructor_dummy_arguments
        n_args = len(constructor_args)

        if n_args == 0:
            code = "(void)"
        elif n_args == 1:
            arg, arg_type = constructor_args[0]
            code = f"(const {arg_type} {arg})"
        else:
            code = "("
            for j, arg in enumerate(constructor_args):
                code += f"\n{INDENT*5}const {arg_type} {arg}"
                if j < n_args - 1:
                    code += ","
            code += ")"

        return code

    def generate_source_code(self, destination, overwrite):
        """
        Generate the C++ source code for the C++ task function's concrete
        TileWrapper class.

        .. todo::
            * I suspect that this is a good candidate for CG-Kit.

        :param destination: Path to folder in which file should be written
        :param overwrite: Overwrite pre-existing file if True
        """
        INDENT = " " * self.indentation

        classname = self.class_name

        path = Path(destination).resolve()
        if not path.is_dir():
            raise RuntimeError(f"{path} is not a folder or does not exist")
        source_filename = path.joinpath(self.source_filename)

        msg = f"Generating C++ Source {source_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        if (not overwrite) and source_filename.exists():
            raise RuntimeError(f"{source_filename} already exists")

        with open(source_filename, "w") as fptr:
            # ----- HEADER INCLUSION
            # Task function's header file
            fptr.write(f'#include "{self.header_filename}"\n')
            fptr.write("\n")

            # Milhoja header files
            fptr.write("#include <Milhoja_Runtime.h>\n")
            fptr.write("#include <Milhoja_RuntimeBackend.h>\n")
            fptr.write("#ifdef DEBUG_RUNTIME\n")
            fptr.write("#include <Milhoja_Logger.h>\n")
            fptr.write("#endif\n")
            fptr.write("\n")

            # ----- STATIC DEFINITIONS
            scratch_vars = self.__scratch_variables
            for arg in sorted(list(scratch_vars)):
                fptr.write(f"void*  {classname}::{arg}_ = nullptr;\n")
            fptr.write("\n")
            fptr.write(f"void {classname}::acquireScratch(void) {{\n")
            if len(scratch_vars) > 0:
                fptr.write(f"{INDENT}const unsigned int  nThreads = ")
                fptr.write("milhoja::Runtime::instance().nMaxThreadsPerTeam();\n")
                fptr.write("\n")
            for arg in sorted(list(scratch_vars)):
                arg_spec = self.__scratch_specification(arg)
                arg_type = arg_spec["type"]
                fptr.write(f"{INDENT}if ({arg}_) {{\n")
                fptr.write(f"{INDENT*2}throw ")
                fptr.write(f'std::logic_error("[{classname}::acquireScratch] ')
                fptr.write(f'{arg}_ scratch already allocated");\n')
                fptr.write(f"{INDENT}}}\n")
                fptr.write("\n")
                fptr.write(f"{INDENT}const std::size_t nBytes = nThreads\n")
                fptr.write(f"{INDENT*5}* {classname}::{arg.upper()}_SIZE_\n")
                fptr.write(f"{INDENT*5}* sizeof({arg_type});\n")
                fptr.write("\n")
                fptr.write(f"{INDENT}milhoja::RuntimeBackend::instance().")
                fptr.write(f"requestCpuMemory(nBytes, &{arg}_);\n")
                fptr.write("\n")
                fptr.write("#ifdef DEBUG_RUNTIME\n")
                fptr.write(f'{INDENT}std::string   msg = "[{classname}::acquireScratch] Acquired"\n')
                fptr.write(f"{INDENT*5}+ std::to_string(nThreads)\n")
                fptr.write(f'{INDENT*5}+ " {arg}_ scratch blocks"\n')
                fptr.write(f"{INDENT}milhoja::Logger::instance().log(msg);\n")
                fptr.write("#endif\n")
            fptr.write("}\n")
            fptr.write("\n")

            fptr.write(f"void {classname}::releaseScratch(void) {{\n")
            for arg in sorted(list(scratch_vars)):
                arg_spec = self.__scratch_specification(arg)
                arg_type = arg_spec["type"]
                fptr.write(f"{INDENT}if (!{arg}_) {{\n")
                fptr.write(f"{INDENT*2}throw ")
                fptr.write(f'std::logic_error("[{classname}::releaseScratch] ')
                fptr.write(f'{arg}_ scratch not allocated");\n')
                fptr.write(f"{INDENT}}}\n")
                fptr.write("\n")
                fptr.write(f"{INDENT}milhoja::RuntimeBackend::instance().")
                fptr.write(f"releaseCpuMemory(&{arg}_);\n")
                fptr.write(f"{INDENT}{arg}_ = nullptr;\n")
                fptr.write("\n")
                fptr.write("#ifdef DEBUG_RUNTIME\n")
                fptr.write(f'{INDENT}std::string   msg = "[{classname}::releaseScratch] ')
                fptr.write(f'Released {arg}_ scratch"\n')
                fptr.write(f"{INDENT}milhoja::Logger::instance().log(msg);\n")
                fptr.write("#endif\n")
            fptr.write("}\n")
            fptr.write("\n")

            # ----- CONSTRUCTOR/DESTRUCTOR
            constructor_args = self._tf_spec.constructor_dummy_arguments
            arg_list = self.__generate_constructor_declaration()
            fptr.write(f"{classname}::{classname}{arg_list}\n")
            fptr.write(f"{INDENT}: milhoja::TileWrapper{{}}")
            fptr.write("\n" if len(constructor_args) == 0 else ",\n")
            for j, (arg, _) in enumerate(constructor_args):
                fptr.write(f"{INDENT}  {arg}_{{{arg}}}")
                fptr.write(",\n" if j < len(constructor_args) - 1 else "\n")
            fptr.write("{\n")
            fptr.write("#ifdef DEBUG_RUNTIME\n")
            fptr.write(f'{INDENT}std::string   msg = "[{classname}] ')
            fptr.write('Creating wrapper object";\n')
            fptr.write(f"{INDENT}milhoja::Logger::instance().log(msg);\n")
            fptr.write("#endif\n")
            fptr.write("}\n")
            fptr.write("\n")

            fptr.write(f"{classname}::~{classname}(void) {{\n")
            fptr.write("#ifdef DEBUG_RUNTIME\n")
            fptr.write(f'{INDENT}std::string   msg = "[~{classname}] ')
            fptr.write('Destroying wrapper object";\n')
            fptr.write(f"{INDENT}milhoja::Logger::instance().log(msg);\n")
            fptr.write("#endif\n")
            fptr.write("}\n")
            fptr.write("\n")

            # ----- CLONE METHOD
            fptr.write("std::unique_ptr<milhoja::TileWrapper> ")
            fptr.write(f"{classname}::clone")
            fptr.write("(std::shared_ptr<milhoja::Tile>&& tileToWrap)")
            fptr.write(" const {\n")
            fptr.write(f"{INDENT}{classname}* ptr = new {classname}")
            if len(constructor_args) == 0:
                fptr.write("{};\n")
            elif len(constructor_args) == 1:
                fptr.write(f"{{{constructor_args[0][0]}_}};\n")
            else:
                for j, (arg, _) in enumerate(constructor_args):
                    fptr.write(f"\n{INDENT*5}{arg}_")
                    if j < len(constructor_args) - 1:
                        fptr.write(",")
                fptr.write("};\n")
            fptr.write("\n")
            fptr.write(f"{INDENT}if (ptr->tile_) {{\n")
            fptr.write(f'{INDENT*2}throw std::logic_error("')
            fptr.write(f'[{classname}::clone] ')
            fptr.write('Internal tile_ member not null");\n')
            fptr.write(f"{INDENT}}}\n")
            fptr.write(f"{INDENT}ptr->tile_ = std::move(tileToWrap);\n")
            fptr.write(f"{INDENT}if (!(ptr->tile_) || tileToWrap) {{\n")
            fptr.write(f'{INDENT*2}throw std::logic_error("')
            fptr.write(f'[{classname}::clone] ')
            fptr.write('Wrapper did not take ownership of tile");\n')
            fptr.write(f"{INDENT}}}\n")
            fptr.write("\n")
            fptr.write(f"{INDENT}return std::unique_ptr<milhoja::TileWrapper>{{ptr}};\n")
            fptr.write("}\n")

    def generate_header_code(self, destination, overwrite):
        """
        Generate the C++ header code for the C++ task function's concrete
        TileWrapper class.

        .. todo::
            * Should scratch variables by correctly typed or all void*?
            * I suspect that this is a good candidate for CG-Kit.

        :param destination: Path to folder in which file should be written
        :param overwrite: Overwrite pre-existing file if True
        """
        INDENT = " " * self.indentation

        classname = self.class_name

        path = Path(destination).resolve()
        if not path.is_dir():
            raise RuntimeError(f"{path} is not a folder or does not exist")
        header_filename = path.joinpath(self.header_filename)
        hdr_macro = f"MILHOJA_GENERATED_{header_filename.stem.upper()}_H__"

        msg = f"Generating C++ Header {header_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        if (not overwrite) and header_filename.exists():
            raise RuntimeError(f"{header_filename} already exists")

        with open(header_filename, "w") as fptr:
            fptr.write(f"#ifndef {hdr_macro}\n")
            fptr.write(f"#define {hdr_macro}\n")
            fptr.write("\n")
            fptr.write("#include <Milhoja_TileWrapper.h>\n")
            fptr.write("\n")

            arg_list = self.__generate_constructor_declaration()

            fptr.write(f"struct {classname} : public milhoja::TileWrapper {{\n")
            fptr.write(f"{INDENT}{classname}{arg_list};\n")
            fptr.write(f"{INDENT}~{classname}(void);\n")
            fptr.write("\n")
            fptr.write(f"{INDENT}{classname}({classname}&)                  = delete;\n")
            fptr.write(f"{INDENT}{classname}(const {classname}&)            = delete;\n")
            fptr.write(f"{INDENT}{classname}({classname}&&)                 = delete;\n")
            fptr.write(f"{INDENT}{classname}& operator=({classname}&)       = delete;\n")
            fptr.write(f"{INDENT}{classname}& operator=(const {classname}&) = delete;\n")
            fptr.write(f"{INDENT}{classname}& operator=({classname}&&)      = delete;\n")
            fptr.write("\n")

            fptr.write(f"{INDENT}std::unique_ptr<milhoja::TileWrapper> ")
            fptr.write("clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) ")
            fptr.write("const override;\n")
            fptr.write("\n")

            constructor_args = self._tf_spec.constructor_dummy_arguments
            for arg, arg_type in constructor_args:
                fptr.write(f"{INDENT}{arg_type}  {arg}_;\n")
            fptr.write("\n")

            fptr.write(f"{INDENT}static void acquireScratch(void);\n")
            fptr.write(f"{INDENT}static void releaseScratch(void);\n")
            fptr.write("\n")
            for arg in sorted(list(self.__scratch_variables)):
                arg_spec = self.__scratch_specification(arg)
                arg_extents = self.__parse_extents_spec(arg_spec["extents"])
                fptr.write(f"{INDENT}constexpr static std::size_t  {arg.upper()}_SIZE_ =")
                if len(arg_extents) == 1:
                    fptr.write(f" {arg_extents[0]};\n")
                else:
                    for j, each in enumerate(arg_extents):
                        if j == 0:
                            fptr.write(f"\n{INDENT*5}  {each}")
                        else:
                            fptr.write(f"\n{INDENT*5}* {each}")
                    fptr.write(";\n")
            fptr.write("\n")
            for arg in sorted(list(self.__scratch_variables)):
                fptr.write(f"{INDENT}static void* {arg}_;\n")
            fptr.write("};\n")
            fptr.write("\n")

            fptr.write("#endif\n")

    def __str__(self):
        json_fname = self.specification_filename
        msg = f"Task Function Specification File\t{json_fname}\n"
        msg += f"TileWrapper C++ Header File\t\t{self.header_filename}\n"
        msg += f"TileWrapper C++ Source File\t\t{self.source_filename}\n"
        msg += f"Indentation length\t\t\t{self.indentation}\n"
        msg += f"Verbosity level\t\t\t\t{self.verbosity_level}"

        return msg
