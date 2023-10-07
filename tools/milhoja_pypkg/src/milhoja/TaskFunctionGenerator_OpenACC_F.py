from pathlib import Path

from . import LOG_LEVEL_BASIC
from . import LOG_LEVEL_BASIC_DEBUG
from . import TaskFunction
from . import AbcCodeGenerator


class TaskFunctionGenerator_OpenACC_F(AbcCodeGenerator):
    """
    A class for generating final, compilable Fortran source code for the task
    function specified by the TaskFunction object given at instantiation.

    .. todo::
        * Should this be able to write with any type of offloading?
    """
    __LOG_TAG = "Milhoja Fortran/OpenACC Task Function"

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

        msg = "Loaded task function specification\n"
        msg += "-" * 80 + "\n"
        msg += str(self)
        self._log(msg, LOG_LEVEL_BASIC_DEBUG)

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

    def generate_header_code(self, destination, overwrite):
        raise LogicError("Fortran task functions do not have a header")

    def generate_source_code(self, destination, overwrite):
        """
        .. todo::
            * This should likely be inserted into the same module as the data
              packet code.  How to get both code generators to insert into the
              same file?  Clearly CG-kit can do this, but where does that outer
              layer of combining code happen?
            * We are presently limited to only offloading around the loop over
              tiles in data packet.  What if we want to launch a kernel within
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
            fptr.write(f"module {module}\n")

            # Setup module & declare interface
            fptr.write(f"{INDENT}implicit none\n")
            fptr.write(f"{INDENT}private\n")
            fptr.write("\n")
            fptr.write(f"{INDENT}public :: {self._tf_spec.name}\n")
            fptr.write("\n")
            fptr.write("contains\n")
            fptr.write("\n")

            # ----- DEFINE TASK FUNCTION SUBROUTINE
            # Begin Subroutine declaration
            dummy_args = self._tf_spec.fortran_dummy_arguments
            fptr.write(f"{INDENT}subroutine {self._tf_spec.name}")
            if len(dummy_args) == 0:
                fptr.write("()\n")
            else:
                fptr.write("( &\n")
                for arg in dummy_args[:-1]:
                    fptr.write(f"{INDENT*5}{arg}, &\n")
                fptr.write(f"{INDENT*5}{dummy_args[-1]} &\n")
                fptr.write(f"{INDENT*3})\n")

            # Boilerplate use statements
            fptr.write(f"{INDENT*2}use iso_c_binding, ONLY : C_PTR\n")
            fptr.write(f"{INDENT*2}use openacc\n")
            fptr.write("\n")
            if self._tf_spec.n_streams > 1:
                fptr.write(f"{INDENT*2}use milhoja_types_mode, ONLY : MILHOJA_INT\n")
                fptr.write("\n")

            # Use in internal subroutines & export for OpenACC
            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
                    interface = \
                        self._tf_spec.subroutine_interface_file(subroutine)
                    assert interface.strip().endswith(".F90")
                    interface = interface.strip().rstrip(".F90")
                    fptr.write(f"{INDENT*2}use {interface}, ONLY : {subroutine}\n")
            fptr.write("\n")

            for node in self._tf_spec.internal_subroutine_graph:
                for subroutine in node:
                    fptr.write(f"{INDENT*2}!$acc routine ({subroutine}) vector\n")
            fptr.write("\n")

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
            points = [
                TaskFunction.TILE_LO, TaskFunction.TILE_HI,
                TaskFunction.TILE_LBOUND, TaskFunction.TILE_UBOUND
            ]
            for arg in self._tf_spec.dummy_arguments:
                spec = self._tf_spec.argument_specification(arg)
                if spec["source"] == TaskFunction.EXTERNAL_ARGUMENT:
                    arg_type = spec["type"]
                    dimension = len(spec["extents"])
                    if dimension == 0:
                        fptr.write(f"{INDENT*2}{arg_type}, intent(IN) :: {arg}_d\n")
                    else:
                        msg = "No test case for non-scalar externals"
                        raise NotImplementedError(msg)
                elif spec["source"] in points:
                    fptr.write(f"{INDENT*2}integer, intent(IN) :: {arg}_d(:, :)\n")
                elif spec["source"] == TaskFunction.TILE_DELTAS:
                    fptr.write(f"{INDENT*2}real, intent(IN) :: {arg}_d(:, :)\n")
                elif spec["source"] == TaskFunction.GRID_DATA_ARGUMENT:
                    if arg in self._tf_spec.tile_in_arguments:
                        intent = "IN"
                    elif arg in self._tf_spec.tile_in_out_arguments:
                        intent = "INOUT"
                    elif arg in self._tf_spec.tile_out_arguments:
                        intent = "OUT"
                    else:
                        raise LogicError("Unknown grid data variable class")
                    fptr.write(f"{INDENT*2}real, intent({intent}) :: {arg}_d(:, :, :, :, :)\n")
                elif spec["source"] == TaskFunction.SCRATCH_ARGUMENT:
                    arg_type = spec["type"]
                    dimension = len(self.__parse_extents_spec(spec["extents"]))
                    assert dimension > 0
                    tmp = [":" for _ in range(dimension + 1)]
                    array = "(" + ", ".join(tmp) + ")"
                    fptr.write(f"{INDENT*2}{arg_type}, intent(OUT) :: {arg}_d{array}\n")
                else:
                    raise LogicError(f"{arg} of unknown argument class")

            fptr.write("\n")

            # Boilerplate local variables
            fptr.write(f"{INDENT*2}integer :: n\n")
            fptr.write("\n")
            if self._tf_spec.n_streams > 1:
                fptr.write(f"{INDENT*2}integer(MILHOJA_INT) :: MH_idx\n")
                fptr.write(f"{INDENT*2}integer(MILHOJA_INT) :: MH_ierr\n")
                fptr.write("\n")

            # Begin OpenACC data region
            device_args = self._tf_spec.fortran_device_dummy_arguments
            if len(device_args) == 0:
                raise NotImplementedError("No test case for no arguments")
            fptr.write(f"{INDENT*2}!$acc data &\n")
            fptr.write(f"{INDENT*2}!$acc& deviceptr( &\n")
            for arg in device_args[:-1]:
                fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{arg}, &\n")
            fptr.write(f"{INDENT*2}!$acc&{INDENT*2}{device_args[-1]} &\n")
            fptr.write(f"{INDENT*2}!$acc&{INDENT})\n")
            fptr.write("\n")

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
                    fptr.write(f"{INDENT*2}!$acc wait( &\n")
                    # We don't have to wait for the data Q explicitly because
                    # there is only one next kernel launch to be scheduled on
                    # the data queue.
                    wait_for = set(current_queues).difference(set(next_queues))
                    wait_for = sorted(list(wait_for))
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
                    assert len(next_queues) > 1
                    for queue in current_queues[-1]:
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
                    for argument in actual_args[:-1]:
                        fptr.write(f"{INDENT*5}{argument}_d, &\n")
                    fptr.write(f"{INDENT*5}{actual_args[-1]}_d &\n")
                    fptr.write(f"{INDENT*5})\n")
                    fptr.write(f"{INDENT*2}end do\n")
                    fptr.write(f"{INDENT*2}!$acc end parallel loop\n")
                    fptr.write("\n")

            # Final wait for task function
            fptr.write(f"{INDENT*2}!$acc wait(dataQ_h)\n")
            fptr.write("\n")

            # Release all extra asynchronous queues after final wait
            release_function = self._tf_spec.release_stream_C_function
            for idx in range(2, self._tf_spec.n_streams+1):
                fptr.write(f"{INDENT*2}MH_idx = INT({idx}, kind=MILHOJA_INT)\n")
                fptr.write(f"{INDENT*2}MH_ierr = {release_function}(C_packet_h, MH_idx)\n")
                fptr.write(f"{INDENT*2}if (MH_ierr /= MILHOJA_SUCCESS) then\n")
                msg = f"Unable to release extra OpenACC async queue {idx}"
                fptr.write(f'{INDENT*3}write(*,*) "[{self._tf_spec.name}] {msg}"\n')
                fptr.write(f"{INDENT*3}STOP\n")
                fptr.write(f"{INDENT*2}end if\n")
                fptr.write("\n")

            # End OpenACC data region
            fptr.write(f"{INDENT*2}!$acc end data\n")

            # End subroutine declaration
            fptr.write(f"{INDENT}end subroutine {self._tf_spec.name}\n")
            fptr.write("\n")

            # End module declaration
            fptr.write(f"end module {module}\n")

    def __str__(self):
        msg = f"Task Function Specification File\t{self.specification_filename}\n"
        msg += f"Fortran/OpenCC Module File\t\t{self.source_filename}\n"
        msg += f"Indentation length\t\t\t{self.indentation}\n"
        msg += f"Verbosity level\t\t\t\t{self.verbosity_level}"

        return msg
