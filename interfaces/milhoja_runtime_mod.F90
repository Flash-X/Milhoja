!> A module in the Milhoja Fortran/C++ interoperability layer that provides
!! calling code with a high-level interface for interacting with the runtime.
!!
!! @todo Add in executeTask routines for other thread team configurations.

#include "milhoja_interface_error_codes.h"

module milhoja_runtime_mod
    use milhoja_types_mod, ONLY : MILHOJA_INT, &
                                  MILHOJA_SIZE_T

    implicit none
    private

    !!!!!----- PUBLIC INTERFACE
    public :: milhoja_runtime_init
    public :: milhoja_runtime_finalize
    public :: milhoja_runtime_taskFunction
    public :: milhoja_runtime_executeTasks_Cpu

    !!!!!----- FORTRAN INTERFACES TO MILHOJA FUNCTION POINTERS
    abstract interface
        !> Fortran interface of the runtime's task function.
        subroutine milhoja_runtime_taskFunction(C_tId, C_dataItemPtr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_tId
            type(C_PTR),          intent(IN), value :: C_dataItemPtr
        end subroutine milhoja_runtime_taskFunction
    end interface

    !!!!!----- INTERFACES TO C-LINKAGE C++ FUNCTIONS
    ! The C-to-Fortran interoperability layer
    interface
        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_runtime_init_c(globalCommF,                      &
                                        nThreadTeams, nThreadsPerTeam,    &
                                        nStreams,                         &
                                        nBytesInMemoryPools) result(ierr) &
                                        bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_SIZE_T
            implicit none
            integer(MILHOJA_INT),    intent(IN), value :: globalCommF
            integer(MILHOJA_INT),    intent(IN), value :: nThreadTeams
            integer(MILHOJA_INT),    intent(IN), value :: nThreadsPerTeam
            integer(MILHOJA_INT),    intent(IN), value :: nStreams
            integer(MILHOJA_SIZE_T), intent(IN), value :: nBytesInMemoryPools
            integer(MILHOJA_INT)                       :: ierr
        end function milhoja_runtime_init_c

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_runtime_finalize_c() result(ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT) :: ierr
        end function milhoja_runtime_finalize_c

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_runtime_execute_tasks_cpu_c(C_taskFunction,            &
                                                     C_nThreads) result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_taskFunction 
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_execute_tasks_cpu_c
    end interface

contains

    !> Initialize the runtime.  This assumes that MPI has already been
    !! initialized by the calling code.
    !!
    !! @todo Does this unit or the grid unit need to be initialized
    !!       first?  If so, document here and in Grid.
    !!
    !! @param globalCommF         The Fortran version of the MPI communicator
    !!                            that Milhoja should use
    !! @param nThreadTeams        The number of thread teams to create
    !! @param nThreadsPerTeam     The number of threads to create in each team
    !! @param nStreams            The number of streams to create
    !! @param nBytesInMemoryPools The size of memory pools that should be
    !!                            eagerly acquired in all memory spaces.  Note
    !!                            the abnormal integer kind.
    !! @param ierr                The milhoja error code
    subroutine milhoja_runtime_init(globalCommF,                   &
                                    nThreadTeams, nThreadsPerTeam, &
                                    nStreams,                      &
                                    nBytesInMemoryPools,           &
                                    ierr)
        integer(MILHOJA_INT),    intent(IN)  :: globalCommF
        integer(MILHOJA_INT),    intent(IN)  :: nThreadTeams
        integer(MILHOJA_INT),    intent(IN)  :: nThreadsPerTeam
        integer(MILHOJA_INT),    intent(IN)  :: nStreams
        integer(MILHOJA_SIZE_T), intent(IN)  :: nBytesInMemoryPools
        integer(MILHOJA_INT),    intent(OUT) :: ierr

        ierr = milhoja_runtime_init_c(globalCommF,                   &
                                      nThreadTeams, nThreadsPerTeam, &
                                      nStreams,                      &
                                      nBytesInMemoryPools)
    end subroutine milhoja_runtime_init

    !> Finalize the runtime.  It is assumed that calling code is responsible for
    !! finalizing MPI and does so *after* calling this routine.
    !!
    !! Calling code should finalize the grid before finalizing the runtime.
    !!
    !! @todo Confirm that grid must be finalized first.
    !!
    !! @param F_ierr    The milhoja error code
    subroutine milhoja_runtime_finalize(F_ierr)
        integer, intent(OUT) :: F_ierr

        integer(MILHOJA_INT) :: C_ierr

        C_ierr = milhoja_runtime_finalize_c()
        F_ierr = INT(C_ierr)
    end subroutine milhoja_runtime_finalize

    !> Execute the given task function using the CPU-only thread team
    !> configuration and with the given number of team threads activated.
    !!
    !! @param F_taskFunction    The task function to execute
    !! @param F_nThreads        The number of threads to activate in team
    !! @param F_ierr            The milhoja error code
    subroutine milhoja_runtime_executeTasks_Cpu(F_taskFunction, F_nThreads, &
                                                F_ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: F_taskFunction
        integer,                                intent(IN)  :: F_nThreads
        integer,                                intent(OUT) :: F_ierr

        type(C_FUNPTR)       :: C_taskFunction
        integer(MILHOJA_INT) :: C_nThreads
        integer(MILHOJA_INT) :: C_ierr

        C_nThreads = INT(F_nThreads, kind=MILHOJA_INT)
        C_taskFunction = C_FUNLOC(F_taskFunction)

        C_ierr = milhoja_runtime_execute_tasks_cpu_c(C_taskFunction, C_nThreads)
        F_ierr = INT(C_ierr)
    end subroutine milhoja_runtime_executeTasks_Cpu

end module milhoja_runtime_mod

