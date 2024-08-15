from pathlib import Path

from .parse_helpers import parse_extents
from . import LogicError
from . import AbcCodeGenerator
from . import TaskFunction
from . import (
    LOG_LEVEL_BASIC, LOG_LEVEL_BASIC_DEBUG, EXTERNAL_ARGUMENT, TILE_LO_ARGUMENT,
    TILE_HI_ARGUMENT, TILE_LBOUND_ARGUMENT, TILE_UBOUND_ARGUMENT,
    TILE_DELTAS_ARGUMENT, GRID_DATA_ARGUMENT, SCRATCH_ARGUMENT, LBOUND_ARGUMENT,
    C2F_TYPE_MAPPING, TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT,
    TILE_LEVEL_ARGUMENT
)


class TaskFunctionGenerator_OpenACC_F(AbcCodeGenerator):
    """
    A class for generating final, compilable Fortran source code for the task
    function specified by the TaskFunction object given at instantiation.

    .. todo::
        * Should this be able to write with any type of offloading?
        * The module file and the top portion of that including the interface
          declarations should likely be a CG-Kit template.  Writing the
          subroutine to be included in the template by CG-kit might stay as is.
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
            TaskFunctionGenerator_OpenACC_F.__LOG_TAG, logger
        )

        msgs_all = [
            "Loaded task function specification",
            "-" * 80,
            f"Specification File\t\t{self.specification_filename}",
            f"Fortran/OpenACC Module File\t{self.source_filename}",
            f"Indentation length\t\t{self.indentation}",
            f"Verbosity level\t\t\t{self.verbosity_level}"
        ]
        for msg in msgs_all:
            self._log(msg, LOG_LEVEL_BASIC_DEBUG)

    def generate_header_code(self, destination, overwrite):
        raise LogicError("Fortran task functions do not have a header")

    def generate_source_code(self, destination, overwrite):
        """
        .. todo::
            * We are presently limited to only offloading around the loop over
              tiles in data packet. What if we want to launch a kernel within
              the loop?  What if we don't want to launch any kernels so that
              the internal subroutine can do its own launching?
            * The extra asynchronous queues should be released as soon as
              possible so that the execution of this doesn't needlessly block
              the execution of other tasks.
            * How to handle Milhoja errors in an internal way that can help
              applications?  Are these errors considered so improbable that the
              error checking here is effectively an assert?
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
            if self._tf_spec.n_streams > 1:
                fptr.write('#include "Milhoja_interface_error_codes.h"\n')
                fptr.write("\n")

            # Begin module declaration
            fptr.writelines([
                f"module {module}\n",
                # Setup module & declare interface
                f"{INDENT}implicit none\n",
                f"{INDENT}private\n\n",
                f"{INDENT}public :: {self._tf_spec.function_name}\n",
                f"{INDENT}public :: {self._tf_spec.cpp2c_layer_name}\n\n",
                f"{INDENT}interface\n{INDENT*2}",
                "!> C++ task function that TimeAdvance passes to Orchestration unit\n",
                f"{INDENT*2}subroutine {self._tf_spec.cpp2c_layer_name}",
                "(C_tId, C_dataItemPtr) &\n",
                f'{INDENT*4}bind(c, name="{self._tf_spec.cpp2c_layer_name}")\n',
                f"{INDENT*3}use iso_c_binding, ONLY : C_PTR\n",
                f"{INDENT*3}use milhoja_types_mod, ONLY : MILHOJA_INT\n",
                f"{INDENT*3}integer(MILHOJA_INT), intent(IN), value :: C_tId\n",
                f"{INDENT*3}type(C_PTR), intent(IN), value :: C_dataItemPtr\n",
                f"{INDENT*2}end subroutine {self._tf_spec.cpp2c_layer_name}\n",
                f"{INDENT}end interface\n\ncontains\n\n"
            ])

            # ----- DEFINE TASK FUNCTION SUBROUTINE
            # Begin Subroutine declaration
            dummy_args = self._tf_spec.fortran_dummy_arguments
            fptr.write(f"{INDENT}subroutine {self._tf_spec.function_name}")
            dummy_arg_str = \
                f"( &\n{INDENT*5}" + f", &\n{INDENT*5}".join(dummy_args) + f" &\n{INDENT*3})\n"
            dummy_arg_str = "()\n" if len(dummy_args) == 0 else dummy_arg_str
            fptr.write(dummy_arg_str)

            if self._tf_spec.n_streams > 1:
                data_item_mod = self._tf_spec.data_item_module_name
                release = self._tf_spec.release_stream_C_function
                fptr.write(f"{INDENT*2}use {data_item_mod}, ONLY : {release}\n")

            # Boilerplate use statements
            fptr.write(f"{INDENT*2}use iso_c_binding, ONLY : C_PTR\n")
            fptr.write(f"{INDENT*2}use openacc\n\n")
            if self._tf_spec.n_streams > 1:
                fptr.write(f"{INDENT*2}use milhoja_types_mod, ONLY : MILHOJA_INT\n\n")

            offloading = []
            # Use in internal subroutines & export for OpenACC
            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
                    interface = \
                        self._tf_spec.subroutine_interface_file(subroutine).strip()
                    assert interface.endswith(".F90")
                    interface = interface.rstrip(".F90")
                    fptr.write(f"{INDENT*2}use {interface}, ONLY : {subroutine}\n")
                    offloading.append(f"{INDENT*2}!$acc routine ({subroutine}) vector\n")
            fptr.writelines(["\n", *offloading, "\n"])
            # No implicit variables
            fptr.write(f"{INDENT*2}implicit none\n\n")

            # Milhoja-internal host-side variables
            #
            # These are not included in the TF dummy arguments
            # TODO: Should this get fortran_host_dummy_arguments and only
            # write lines for arguments in the result?
            fptr.write(f"{INDENT*2}type(C_PTR), intent(IN) :: C_packet_h\n")
            fptr.write(f"{INDENT*2}integer(kind=acc_handle_kind), intent(IN) :: dataQ_h\n")
            for i in range(2, self._tf_spec.n_streams+1):
                queue = f"queue{i}_h"
                fptr.write(f"{INDENT*2}integer(kind=acc_handle_kind), intent(IN) :: {queue}\n")
            fptr.write(f"{INDENT*2}integer, intent(IN) :: nTiles_d\n")

            # Generation-time argument definitions
            points = {
                TILE_LO_ARGUMENT, TILE_HI_ARGUMENT, TILE_LBOUND_ARGUMENT,
                TILE_UBOUND_ARGUMENT, LBOUND_ARGUMENT
            }
            bounds = {TILE_INTERIOR_ARGUMENT, TILE_ARRAY_BOUNDS_ARGUMENT}

            for arg in self._tf_spec.dummy_arguments:
                spec = self._tf_spec.argument_specification(arg)
                src = spec["source"]
                if src == EXTERNAL_ARGUMENT:
                    extents = spec["extents"]
                    if extents != "()":
                        msg = "No test case for non-scalar externals"
                        raise NotImplementedError(msg)

                    # is this okay? Should we fail if there is no type mapping?
                    arg_type = C2F_TYPE_MAPPING.get(spec["type"], spec["type"])
                    fptr.write(f"{INDENT*2}{arg_type}, intent(IN) :: {arg}_d\n")

                elif src in points:
                    fptr.write(f"{INDENT*2}integer, intent(IN) :: {arg}_d(:, :)\n")

                elif src == TILE_DELTAS_ARGUMENT:
                    fptr.write(f"{INDENT*2}real, intent(IN) :: {arg}_d(:, :)\n")

                elif src in bounds:
                    fptr.write(f"{INDENT*2}integer, intent(IN) :: {arg}_d(:, :, :)\n")

                elif src == TILE_LEVEL_ARGUMENT:
                    fptr.write(f"{INDENT*2}integer, intent(IN) :: {arg}_d(:, :)\n")

                elif src == GRID_DATA_ARGUMENT:
                    if arg in self._tf_spec.tile_in_arguments:
                        intent = "IN"
                    elif arg in self._tf_spec.tile_in_out_arguments:
                        intent = "INOUT"
                    elif arg in self._tf_spec.tile_out_arguments:
                        intent = "OUT"
                    else:
                        raise LogicError("Unknown grid data variable class")

                    fptr.write(f"{INDENT*2}real, intent({intent}) :: {arg}_d(:, :, :, :, :)\n")

                elif src == SCRATCH_ARGUMENT:
                    arg_type = spec["type"]
                    dimension = len(parse_extents(spec["extents"]))
                    assert dimension > 0
                    tmp = [":" for _ in range(dimension + 1)]
                    array = "(" + ", ".join(tmp) + ")"
                    fptr.write(f"{INDENT*2}{arg_type}, intent(INOUT) :: {arg}_d{array}\n")

                else:
                    raise LogicError(f"{arg} of unknown argument class")

            fptr.write("\n")

            # Boilerplate local variables
            fptr.write(f"{INDENT*2}integer :: n\n\n")
            if self._tf_spec.n_streams > 1:
                fptr.write(f"{INDENT*2}integer(MILHOJA_INT) :: MH_idx\n")
                fptr.write(f"{INDENT*2}integer(MILHOJA_INT) :: MH_ierr\n\n")

            # Begin OpenACC data region
            device_args = self._tf_spec.fortran_device_dummy_arguments
            if len(device_args) == 0:
                raise NotImplementedError("No test case for no arguments")

            fptr.write(f"{INDENT*2}!$acc data &\n")
            fptr.write(f"{INDENT*2}!$acc& deviceptr( &\n")
            for arg in device_args[:-1]:
                fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{arg}, &\n")
            fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{device_args[-1]} &\n")
            fptr.write(f"{INDENT*2}!$acc&{INDENT})\n\n")

            # Implement internal subroutine call graph with OpenACC offloading.
            # Data packet sent on dataQ_h
            current_queues = ["dataQ_h"]

            for node in self._tf_spec.internal_subroutine_graph:
                # Insert waits if needed before next round of kernel launches
                extras = [f"queue{i}_h" for i in range(2, len(node) + 1)]
                next_queues = ["dataQ_h"] + extras
                if (current_queues == ["dataQ_h"]) and \
                        (len(next_queues) > 1):
                    fptr.write(f"{INDENT*2}!$acc wait(dataQ_h)\n")
                    fptr.write("\n")
                elif (len(current_queues) > 1) and \
                        (next_queues == ["dataQ_h"]):
                    # We don't have to wait for the data Q explicitly because
                    # there is only one next kernel launch to be scheduled on
                    # the data queue.
                    wait_for = set(current_queues).difference(set(next_queues))
                    wait_for = sorted(list(wait_for))
                    fptr.write(f"{INDENT*2}!$acc wait( &\n")
                    for queue in wait_for[:-1]:
                        fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{queue}, &\n")
                    fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{wait_for[-1]} &\n")
                    fptr.write(f"{INDENT*2}!$acc& )\n")
                    fptr.write("\n")
                elif len(current_queues) > 1:
                    # Insert a barrier on all queues.
                    #
                    # This is necessary even if current and next queues are the
                    # same and we are using more than one stream.  For now we
                    # do *not* assume that the kernel launches can flow from
                    # one launch to the next on these streams.
                    raise NotImplementedError("Not tested yet")
                    assert len(next_queues) > 1
                    for queue in current_queues[:-1]:
                        fptr.write(f"{INDENT*2}!$acc& {queue}, &\n")
                    fptr.write(f"{INDENT*2}!$acc& {current_queues[-1]} &\n")
                    fptr.write(f"{INDENT*2}!$acc& )\n")
                    fptr.write("\n")
                else:
                    # Next kernel launch on same stream => no need to wait
                    assert current_queues == ["dataQ_h"]
                    assert current_queues == next_queues

                current_queues = next_queues.copy()
                assert len(current_queues) == len(node)
                for subroutine, queue in zip(node, current_queues):
                    fptr.write(f"{INDENT*2}!$acc parallel loop gang default(none) &\n")
                    fptr.write(f"{INDENT*2}!$acc& async({queue})\n")
                    fptr.write(f"{INDENT*2}do n = 1, nTiles_d\n")
                    fptr.write(f"{INDENT*3}CALL {subroutine}( &\n")
                    actual_args = \
                        self._tf_spec.subroutine_actual_arguments(subroutine)
                    arg_list = []
                    for argument in actual_args:
                        spec = self._tf_spec.argument_specification(argument)
                        extents = ""
                        if spec["source"] in points:
                            extents = "(:, n)"
                        elif spec["source"] == TILE_DELTAS_ARGUMENT:
                            extents = "(:, n)"
                        elif spec["source"] in bounds:
                            extents = "(:, :, n)"
                        elif spec["source"] == GRID_DATA_ARGUMENT:
                            extents = "(:, :, :, :, n)"
                        elif spec["source"] == SCRATCH_ARGUMENT:
                            dimension = len(parse_extents(spec["extents"]))
                            tmp = [":" for _ in range(dimension)]
                            extents = "(" + ", ".join(tmp) + ", n)"
                        arg_list.append(f"{INDENT*5}{argument}_d{extents}")
                    fptr.write(", &\n".join(arg_list) + " &\n")
                    fptr.write(f"{INDENT*5})\n")
                    fptr.write(f"{INDENT*2}end do\n")
                    fptr.write(f"{INDENT*2}!$acc end parallel loop\n")
                    fptr.write("\n")

            # Final wait for task function
            fptr.write(f"{INDENT*2}!$acc wait( &\n")
            for queue in current_queues[:-1]:
                fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{queue}, &\n")
            fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{current_queues[-1]} &\n")
            fptr.write(f"{INDENT*2}!$acc&{INDENT})\n\n")

            # Release all extra asynchronous queues after final wait
            for idx in range(2, self._tf_spec.n_streams+1):
                release_function = self._tf_spec.release_stream_C_function
                fptr.write(f"{INDENT*2}MH_idx = INT({idx}, kind=MILHOJA_INT)\n")
                fptr.write(f"{INDENT*2}MH_ierr = {release_function}(C_packet_h, MH_idx)\n")
                fptr.write(f"{INDENT*2}if (MH_ierr /= MILHOJA_SUCCESS) then\n")
                msg = f"Unable to release extra OpenACC async queue {idx}"
                fptr.write(f'{INDENT*3}write(*,*) "[{self._tf_spec.name}] {msg}"\n')
                fptr.write(f"{INDENT*3}STOP\n")
                fptr.write(f"{INDENT*2}end if\n\n")

            # End OpenACC data region
            fptr.write(f"{INDENT*2}!$acc end data\n")
            # End subroutine declaration
            fptr.write(f"{INDENT}end subroutine {self._tf_spec.function_name}\n")
            fptr.write("\n")
            # End module declaration
            fptr.write(f"end module {module}\n\n")
