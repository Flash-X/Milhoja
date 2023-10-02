import json

from pathlib import Path

from . import LOG_LEVEL_BASIC
from . import LOG_LEVEL_BASIC_DEBUG
from . import TaskFunction
from . import AbcCodeGenerator


class TileWrapperGenerator_cpp(AbcCodeGenerator):
    """
    A class for generating final, compilable C++ header and source code that
    defines a Milhoja_TileWrapper derived class for constructing the data item
    passed to the task function specified by the specification object given at
    instantiation.
    """
    __LOG_TAG = "Milhoja C++ Tile Wrapper"

    # ----- INSTANTIATION CLASS METHODS
    @classmethod
    def from_json(
            cls,
            tf_spec_json_filename,
            log_level=LOG_LEVEL_BASIC,
            indent=4
            ):
        """
        Instantiate an object and initialize it with the contents of the given
        JSON-format file, which contains all configuration information needed
        to generate a derived Milhoja_TileWrapper class for use with the task
        function specified by the JSON file.

        See the constructor's documentation for more information.

        :param tf_spec_json_filename: Name of the JSON-format file
        :type  tf_spec_json_filename: str
        :param logger: Object for logging code generation details
        :type  logger: CodeGenerationLogger or a class derived from that class
        :param indent: The number of spaces used to define the tab to be used
            in both generated files.
        :type  indent: non-negative int, optional
        :return: The generator object ready for use
        :rtype: TileWrapperGenerator_cpp
        """
        json_fname = Path(tf_spec_json_filename).resolve()
        if not json_fname.is_file():
            raise ValueError(f"{json_fname} does not exist or is not a file")

        # TODO: Instead of using .from_* classmethods, the constructor of the
        # TF specification class could guess the format by looking at the file
        # extension.
        with open(json_fname, "r") as fptr:
            tf_spec = json.load(fptr)

        tf_spec_new = TaskFunction.from_milhoja_json(json_fname)

        # TODO: Once we have a class that wraps the task function
        # specification, we should instantiate it here with its .to_json
        # classmethod and pass it to the constructor.  tf_spec is presently the
        # standin for that object.
        generator = TileWrapperGenerator_cpp(
                        tf_spec,
                        tf_spec_new,
                        log_level,
                        indent
                    )

#        msg = f"Created code generator from JSON file {tf_spec_json_filename}"
#        logger.log(msg, LOG_LEVEL_BASIC_DEBUG)

        return generator

    def __init__(
            self,
            tf_spec,
            tf_spec_new,
            log_level,
            indent
            ):
        """
        Construct an object for use with the task function specified by the
        given specification object.

        It is intended that all codes that need this generator instatiate
        generator objects via the .from_* classmethods rather than using this
        constructor directly.

        The basename of the header file is adopted as the name of the
        TileWrapper derived class.

        :param tf_spec: The XXX task function specification object
        :type  tf_spec: XXX
        :param tf_spec_filename: Name of the task function specification file
        :type  tf_spec_filename: str
        :param logger: Object for logging code generation details
        :type  logger: CodeGenerationLogger or a class derived from that class
        :param indent: The number of spaces used to define the tab to be used
            in both generated files.
        :type  indent: non-negative int
        """
        outputs = tf_spec_new.output_filenames
        header_filename = outputs[TaskFunction.DATA_ITEM_KEY]["header"]
        source_filename = outputs[TaskFunction.DATA_ITEM_KEY]["source"]

        super().__init__(
            tf_spec_new,
            header_filename, source_filename,
            TileWrapperGenerator_cpp.__LOG_TAG, log_level,
            indent
        )

        # ----- STORE ARGUMENTS
        self.__tf_spec = tf_spec["task_function"]

        self.__class_name = Path(self.header_filename).stem

        # ----- SANITY CHECK ARGUMENTS
        # Since there could be no file at instantiation, but a file could
        # appear before calling a generate method, we don't check file
        # existence here.

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
        # TODO: The content here is likely generic to all code generators and
        # should likely be made available in a constants.py file in the
        # package.
        self.__TILE_METADATA_LUT = {"tile_lo":     ("const milhoja::IntVect",  "tileDesc->lo()"),
                                    "tile_hi":     ("const milhoja::IntVect",  "tileDesc->hi()"),
                                    "tile_deltas": ("const milhoja::RealVect", "tileDesc->deltas()")}

        self.__TILE_DATA_ARRAY_TYPES = ["milhoja::FArray1D",
                                        "milhoja::FArray2D",
                                        "milhoja::FArray3D",
                                        "milhoja::FArray4D"]
        self.__MIN_DATA_ARRAY_DIM = 1
        self.__MAX_DATA_ARRAY_DIM = len(self.__TILE_DATA_ARRAY_TYPES)

        msg = "Loaded task function specification\n"
        msg += "-" * 80 + "\n"
        msg += str(self)
        self._log(msg, LOG_LEVEL_BASIC_DEBUG)

    @property
    def __external_arguments(self):
        """
        TODO: This seems exactly like something that should be in the task
        function specification class.
        """
        arg_specs_all = self.__tf_spec["argument_specifications"]

        external_all = []
        for arg, arg_spec in arg_specs_all.items():
            if arg_spec["source"].lower() == "external":
                external_all.append((arg, arg_spec["type"]))

        return external_all

    @property
    def __scratch_arguments(self):
        """
        TODO: This seems exactly like something that should be in the task
        function specification class.
        """
        arg_specs_all = self.__tf_spec["argument_specifications"]

        # Scratch requested by application
        scratch_all = []
        for arg, arg_spec in arg_specs_all.items():
            if arg_spec["source"].lower() == "scratch":
                scratch_all.append((arg, arg_spec["type"], arg_spec["extents"]))

        # Internal scratch required by Milhoja
        for arg, arg_spec in arg_specs_all.items():
            if arg_spec["source"].lower() == "tile_cellvolumes":
                # TODO: How to get extents and dimensionality of problem as numbers here?
                name = "MH_INTERNAL_cellVolumes"
                scratch_type = "milhoja::Real"
                extents = "(18, 18, 18)"
                scratch_all.append((name, scratch_type, extents))

        return scratch_all

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

    def __generate_constructor_arg_list(self, external_all):
        """
        """
        INDENT = " " * self.indentation

        n_external = len(external_all)

        processor = self._tf_spec_new.processor

        if n_external == 0:
            arg_list = "(void)"
        elif n_external == 1:
            arg_type = external_all[0][1]
            if (arg_type.lower() == "real") and (processor.lower() == "cpu"):
                arg_type = "milhoja::Real"
            arg_list = f"(const {arg_type} {external_all[0][0]})"
        else:
            arg_list = ""
            for j, (arg, arg_type) in enumerate(external_all):
                if (arg_type.lower() == "real") and (processor.lower() == "cpu"):
                    arg_type = "milhoja::real"
                arg_list += f"\n{INDENT*5}const {arg_type} {arg}"
                if j < n_external - 1:
                    arg_list += ","
            arg_list += ")"

        return arg_list

    def generate_source_code(self, destination, overwrite):
        """Generate the C++ source code"""
        INDENT = " " * self.indentation

        path = Path(destination).resolve()
        if not path.is_dir():
            raise ValueError(f"{path} is not a folder or does not exist")
        source_filename = path.joinpath(self.source_filename)

        msg = f"Generating C++ Source {source_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        classname = self.__class_name

        external_all = self.__external_arguments
        scratch_all = self.__scratch_arguments
        n_external = len(external_all)

        processor = self._tf_spec_new.processor

        if (not overwrite) and source_filename.exists():
            raise ValueError(f"{source_filename} already exists")

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
            for arg, _, _ in scratch_all:
                fptr.write(f"void*  {classname}::{arg}_ = nullptr;\n")
            fptr.write("\n")
            fptr.write(f"void {classname}::acquireScratch(void) {{\n")
            fptr.write(f"{INDENT}const unsigned int  nThreads = ")
            fptr.write("milhoja::Runtime::instance().nMaxThreadsPerTeam();\n")
            fptr.write("\n")
            for arg, arg_type, _ in scratch_all:
                if (arg_type.lower() == "real") and (processor.lower() == "cpu"):
                    arg_type = "milhoja::Real"

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
            for arg, arg_type, _ in scratch_all:
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
            arg_list = self.__generate_constructor_arg_list(external_all)
            fptr.write(f"{classname}::{classname}{arg_list}\n")
            fptr.write(f"{INDENT}: milhoja::TileWrapper{{}}")
            fptr.write("\n" if n_external == 0 else ",\n")
            for j, (arg, _) in enumerate(external_all):
                fptr.write(f"{INDENT}  {arg}_{{{arg}}}")
                fptr.write(",\n" if j < n_external - 1 else "\n")
            fptr.write("{\n")
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
            if n_external == 0:
                fptr.write("{};\n")
            elif n_external == 1:
                fptr.write(f"{{{external_all[0][0]}_}};\n")
            else:
                for j, (arg, _) in enumerate(external_all):
                    fptr.write(f"\n{INDENT*5}{arg}_")
                    if j < n_external - 1:
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
        """Generate the C++ header code"""
        INDENT = " " * self.indentation

        path = Path(destination).resolve()
        if not path.is_dir():
            raise ValueError(f"{path} is not a folder or does not exist")
        header_filename = path.joinpath(self.header_filename)

        msg = f"Generating C++ Header {header_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        hdr_macro = f"MILHOJA_GENERATED_{self.__class_name.upper()}_H__"

        external_all = self.__external_arguments
        scratch_all = self.__scratch_arguments

        processor = self._tf_spec_new.processor

        if (not overwrite) and header_filename.exists():
            raise ValueError(f"{header_filename} already exists")

        with open(header_filename, "w") as fptr:
            fptr.write(f"#ifndef {hdr_macro}\n")
            fptr.write(f"#define {hdr_macro}\n")
            fptr.write("\n")
            fptr.write("#include <Milhoja_TileWrapper.h>\n")
            fptr.write("\n")

            fptr.write(f"struct {self.__class_name} : public milhoja::TileWrapper {{\n")
            arg_list = self.__generate_constructor_arg_list(external_all)
            fptr.write(f"{INDENT}{self.__class_name}{arg_list};\n")
            fptr.write(f"{INDENT}~{self.__class_name}(void);\n")
            fptr.write("\n")
            fptr.write(f"{INDENT}{self.__class_name}({self.__class_name}&)                  = delete;\n")
            fptr.write(f"{INDENT}{self.__class_name}(const {self.__class_name}&)            = delete;\n")
            fptr.write(f"{INDENT}{self.__class_name}({self.__class_name}&&)                 = delete;\n")
            fptr.write(f"{INDENT}{self.__class_name}& operator=({self.__class_name}&)       = delete;\n")
            fptr.write(f"{INDENT}{self.__class_name}& operator=(const {self.__class_name}&) = delete;\n")
            fptr.write(f"{INDENT}{self.__class_name}& operator=({self.__class_name}&&)      = delete;\n")
            fptr.write("\n")

            fptr.write(f"{INDENT}std::unique_ptr<milhoja::TileWrapper> ")
            fptr.write("clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) ")
            fptr.write("const override;\n")
            fptr.write("\n")

            for arg, arg_type in external_all:
                if (arg_type.lower()) == "real" and (processor.lower() == "cpu"):
                    fptr.write(f"{INDENT}milhoja::Real  {arg}_;\n")
                else:
                    fptr.write(f"{INDENT}{arg_type}  {arg}_;\n")
            fptr.write("\n")

            # TODO: Should we only declare & define these if scratch is needed?
            fptr.write(f"{INDENT}static void acquireScratch(void);\n")
            fptr.write(f"{INDENT}static void releaseScratch(void);\n")
            fptr.write("\n")
            for arg, _, extents in scratch_all:
                arg_extents = self.__parse_extents_spec(extents)
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
            for arg, _, _ in scratch_all:
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
