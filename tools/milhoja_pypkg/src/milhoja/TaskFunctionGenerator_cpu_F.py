from pathlib import Path

from .parse_helpers import parse_extents
from . import LogicError
from . import AbcCodeGenerator
from . import TaskFunction
from . import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG, EXTERNAL_ARGUMENT,
    TILE_LO_ARGUMENT, TILE_HI_ARGUMENT, TILE_LBOUND_ARGUMENT,
    TILE_UBOUND_ARGUMENT, TILE_DELTAS_ARGUMENT, GRID_DATA_ARGUMENT,
    SCRATCH_ARGUMENT, LBOUND_ARGUMENT, C2F_TYPE_MAPPING,
    TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT, TILE_LEVEL_ARGUMENT,
    VERBATIM_ARGUMENT,
)


class TaskFunctionGenerator_cpu_F(AbcCodeGenerator):
    """
    A class for generating final, compilable Fortran source code for the task
    function specified by the TaskFunction object given at instantiation.

    .. todo::
        * Should this be able to write with any type of offloading?
        * The module file and the top portion of that including the interface
          declarations should likely be a CG-Kit template. Writing the
          subroutine to be included in the template by CG-kit might
          stay as is.
    """
    __LOG_TAG = "Milhoja Fortran/OpenACC Task Function"

    def __init__(self, tf_spec, indent, logger):
        """
        Construct an object for use with the task function specified by the
        given specification object.

        :param tf_spec: Specification object derived from TaskFunction
        :param log_level: Milhoja level to use for logging generation
        :param logger: Concrete logger derived from AbcLogger
        """
        if not isinstance(tf_spec, TaskFunction):
            raise TypeError("Given tf_spec not derived from TaskFunction")

        outputs = tf_spec.output_filenames
        header_filename = None
        source_filename = outputs[TaskFunction.FORTRAN_TF_KEY]["source"]

        super().__init__(
            tf_spec, header_filename, source_filename, indent,
            TaskFunctionGenerator_cpu_F.__LOG_TAG, logger
        )

        msgs_all = [
            "Loaded task function specification",
            "-" * 80,
            f"Specification File\t\t{self.specification_filename}",
            f"Fortran Module File\t{self.source_filename}",
            f"Indentation length\t\t{self.indentation}",
            f"Verbosity level\t\t\t{self.verbosity_level}"
        ]
        for msg in msgs_all:
            self._log(msg, LOG_LEVEL_BASIC_DEBUG)

    def generate_header_code(self, destination, overwrite):
        """
        Generates header code for a TaskFunction based on the task function
        spec. There is no header file for task functions so this raises
        a LogicError.
        """
        raise LogicError("Fortran task functions do not have a header")

    def generate_source_code(self, destination, overwrite):
        """
        Generate the source code for a task function based on the
        task function specification.

        .. todo::
            * We are presently limited to only offloading around the loop over
              tiles in data packet.  What if we want to launch a kernel within
              the loop?  What if we don't want to launch any kernels so that
              the internal subroutine can do its own launching?
            * The extra asynchronous queues should be released as soon as
              possible so that the execution of this doesn't needlessly block
              the execution of other tasks.
            * How to handle Milhoja errors in an internal way that can help
              applications? Are these errors considered so improbable that the
              error checking here is effectively an assert?

        :param destination: The destination path
        :param overwrite: Whether or not to overwrite any file that might be
                          at the destination.
        """
        INDENT = " " * self.indentation

        dst = Path(destination).resolve()
        if not dst.is_dir():
            raise RuntimeError(f"{dst} is not a folder or does not exist")
        source_filename = dst.joinpath(self.source_filename)

        msg = f"Generating Fortran Source {source_filename}"
        self._log(msg, LOG_LEVEL_BASIC)

        if (not overwrite) and source_filename.exists():
            raise RuntimeError(f"{source_filename} already exists")

        module = self._tf_spec.fortran_module_name

        with open(source_filename, "w") as fptr:
            # ----- DEFINE MODULE & MODULE INTERFACE
            # C Preprocessor includes
            fptr.write('#include "Milhoja.h"\n')

            # todo:: this can probably be pulled out into a separate function
            # Begin module declaration
            fptr.writelines([
                f"module {module}\n",
                # Setup module & declare interface
                f"{INDENT}implicit none\n",
                f"{INDENT}private\n\n",
                f"{INDENT}public :: {self._tf_spec.function_name}\n",
                f"{INDENT}public :: {self._tf_spec.cpp2c_layer_name}\n\n",
                f"{INDENT}interface\n",
                f"{INDENT*2}subroutine {self._tf_spec.cpp2c_layer_name}",
                "(C_threadIndex, C_dataItemPtr) &\n",
                f'{INDENT*2}bind(c, '
                f'name="{self._tf_spec.cpp2c_layer_name}")\n',
                f"{INDENT*3}use iso_c_binding, ONLY : C_PTR\n",
                f"{INDENT*3}use milhoja_types_mod, ONLY : MILHOJA_INT\n",
                f"{INDENT*3}integer(MILHOJA_INT), intent(IN), value :: "
                "C_threadIndex\n",
                f"{INDENT*3}type(C_PTR), intent(IN), value :: "
                "C_dataItemPtr\n",
                f"{INDENT*2}end subroutine "
                f"{self._tf_spec.cpp2c_layer_name}\n",
                f"{INDENT}end interface\n\ncontains\n\n"
            ])

            # ----- DEFINE TASK FUNCTION SUBROUTINE
            # Begin Subroutine declaration
            dummy_args = self._tf_spec.dummy_arguments
            fptr.write(f"{INDENT}subroutine {self._tf_spec.function_name}")
            dummy_arg_str = \
                f"( &\n{INDENT*2}" + f", &\n{INDENT*2}".join(dummy_args) + \
                f" &\n{INDENT})\n"
            dummy_arg_str = "()\n" if len(dummy_args) == 0 else dummy_arg_str
            fptr.write(dummy_arg_str)

            target = ""
            offloading = []
            any_node_with_pointer_args_byName = False
            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
                    interface = \
                        self._tf_spec.subroutine_interface_file(
                            subroutine
                        ).strip()
                    assert interface.endswith(".F90")
                    interface = interface.rstrip(".F90")
                    fptr.write(
                        f"{INDENT*2}use {interface}, ONLY : {subroutine}\n"
                    )

                    # hardwired assumption that these are the only
                    # routine with POINTER arguments that can be
                    # encountered here!
                    has_pointer_args_byName = ("Eos_guardCells" in subroutine or
                                               "Eos_wrapped" in subroutine)
                    any_node_with_pointer_args_byName = (
                        any_node_with_pointer_args_byName or has_pointer_args_byName
                    )
                    if has_pointer_args_byName:
                        target = ", target"

            fptr.writelines(["\n", *offloading, "\n"])
            # No implicit variables
            fptr.write(f"{INDENT*2}implicit none\n\n")

            # Generation-time argument definitions
            points = {
                TILE_LO_ARGUMENT, TILE_HI_ARGUMENT, TILE_LBOUND_ARGUMENT,
                TILE_UBOUND_ARGUMENT, TILE_LEVEL_ARGUMENT
            }
            bounds = {TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT}

            grid_ptr_data = []
            grid_ptr_nullify = []
            eos_ptr = {}
            end_tf = []
            for arg in self._tf_spec.dummy_arguments:
                spec = self._tf_spec.argument_specification(arg)
                src = spec["source"]
                if src == EXTERNAL_ARGUMENT:
                    extents = spec["extents"]
                    if extents != "()":
                        msg = "No test case for non-scalar externals"
                        raise NotImplementedError(msg)

                    # Should we fail if there is no type mapping?
                    arg_type = \
                        C2F_TYPE_MAPPING.get(spec["type"], spec["type"])
                    fptr.write(f"{INDENT*2}{arg_type}, intent(IN) :: {arg}\n")

                elif src in points:
                    fptr.write(f"{INDENT*2}integer, intent(IN) :: {arg}(:)\n")

                elif src == LBOUND_ARGUMENT:
                    fptr.write(f"{INDENT*2}integer, intent(IN) :: {arg}(:)\n")
                    array_spec = \
                        self._tf_spec.argument_specification(spec["array"])
                    if array_spec["source"] == GRID_DATA_ARGUMENT:
                        # We have to set a pointer for subroutines with POINTER dummy arguments -
                        # some "Eos_" routines are known for this, hence the name.
                        eos_ptr[spec["array"]] = \
                            "({0}:,{1}:,{2}:,{3}:)".format(
                                f"{arg}(1)", f"{arg}(2)",
                                f"{arg}(3)", f"{arg}(4)"
                            ) + f" => {spec['array']}"
                        pointerdeclaration = (
                            f"{INDENT*2}real, pointer :: "
                            f"{spec['array']}_ptr(:, :, :, :)\n"
                        )
                        if pointerdeclaration not in grid_ptr_data:
                            grid_ptr_data.extend([pointerdeclaration])
                        grid_ptr_nullify.append(
                            f"{INDENT*2}NULLIFY({spec['array']}_ptr)  ! unnecessary??\n"
                        )
                        if any_node_with_pointer_args_byName:
                            end_tf.extend([f"NULLIFY({spec['array']}_ptr)  ! unnecessary?\n"])

                elif src == TILE_DELTAS_ARGUMENT:
                    fptr.write(f"{INDENT*2}real, intent(IN) :: {arg}(:)\n")

                elif src in bounds:
                    fptr.write(
                        f"{INDENT*2}integer, intent(IN) :: {arg}(:, :)\n"
                    )

                elif src == TILE_LEVEL_ARGUMENT:
                    fptr.write(f"{INDENT*2}integer, intent(IN) :: {arg}\n")

                elif src == GRID_DATA_ARGUMENT:
                    if arg in self._tf_spec.tile_in_arguments:
                        intent = "IN"
                    elif arg in self._tf_spec.tile_in_out_arguments:
                        intent = "INOUT"
                    elif arg in self._tf_spec.tile_out_arguments:
                        intent = "OUT"
                    else:
                        raise LogicError("Unknown grid data variable class")
                    fptr.write(
                        f"{INDENT*2}real, intent({intent}){target}"
                        f" :: {arg}(:, :, :, :)\n"
                    )
                    if arg not in eos_ptr:
                        eos_ptr[arg] = f"(1:,1:,1:,1:) => {arg}\n"  # DEV: or just f" => {arg}\n" ?  To be reviewed! KW
                        pointerdeclaration = (
                            f"{INDENT*2}real, pointer :: "
                            f"{arg}_ptr(:, :, :, :)\n"
                        )
                        if pointerdeclaration not in grid_ptr_data:
                            grid_ptr_data.extend([pointerdeclaration])
#                        ])

                elif src == SCRATCH_ARGUMENT:
                    arg_type = C2F_TYPE_MAPPING.get(
                        spec["type"], spec["type"]
                    )
                    dimension = len(parse_extents(spec["extents"]))
                    assert dimension > 0
                    tmp = [":" for _ in range(dimension)]
                    array = "(" + ", ".join(tmp) + ")"
                    fptr.write(
                        f"{INDENT*2}{arg_type}, intent(INOUT) :: {arg}{array}\n"
                    )

                else:
                    raise LogicError(f"{arg} of unknown argument class")

            fptr.write("\n")
            if any_node_with_pointer_args_byName:
                fptr.write("".join(grid_ptr_data) + "\n")
                fptr.write("".join(grid_ptr_nullify) + "\n")

            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
                    has_pointer_args = ("Eos_guardCells" in subroutine or
                                        "Eos_wrapped" in subroutine)
                    actual_args = \
                        self._tf_spec.subroutine_actual_arguments(subroutine)
                    if has_pointer_args:
                        for key in eos_ptr:
                            if (
                                    key in actual_args
                                    and self._tf_spec.argument_specification(key)["source"] == GRID_DATA_ARGUMENT
                            ):
                                fptr.write(f"{INDENT*2}{key}_ptr{eos_ptr[key]}\n")

                    fptr.write(f"{INDENT*2}CALL {subroutine}( &\n")
                    arg_list = []
                    for argument in actual_args:
                        spec = self._tf_spec.argument_specification(argument)
                        arg = f"{INDENT*3}{argument}"
                        # Eos is weird
                        if (
                            spec["source"] == GRID_DATA_ARGUMENT and
                            has_pointer_args and
                            argument in eos_ptr
                        ):
                            arg += "_ptr"

                        # get the first argument in the tile level array?
                        if argument == TILE_LEVEL_ARGUMENT:
                            arg += "(1) + 1"

                        arg_list.append(arg)
                    fptr.write(", &\n".join(arg_list) + " &\n")
                    fptr.write(f"{INDENT*2})\n")

            # End subroutine declaration
            fptr.write(f"\n{INDENT*2}" + f"{INDENT*2}".join(end_tf))
            fptr.write(
                f"\n{INDENT}end subroutine {self._tf_spec.function_name}\n"
            )
            fptr.write("\n")
            # End module declaration
            fptr.write(f"end module {module}\n\n")
