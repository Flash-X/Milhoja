!> A module in the Milhoja Fortran/C++ interoperability layer that provides
!! calling code with a high-level interface for interacting with the runtime.
!!
!! @todo Add in executeTask routines for other thread team configurations.

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
        function milhoja_runtime_init_c(C_nThreadTeams, C_nThreadsPerTeam,    &
                                        C_nStreams,                         &
                                        C_nBytesInMemoryPools) result(C_ierr) &
                                        bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_SIZE_T
            implicit none
            integer(MILHOJA_INT),    intent(IN), value :: C_nThreadTeams
            integer(MILHOJA_INT),    intent(IN), value :: C_nThreadsPerTeam
            integer(MILHOJA_INT),    intent(IN), value :: C_nStreams
            integer(MILHOJA_SIZE_T), intent(IN), value :: C_nBytesInMemoryPools
            integer(MILHOJA_INT)                       :: C_ierr
        end function milhoja_runtime_init_c

        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_finalize_c() result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT) :: C_ierr
        end function milhoja_runtime_finalize_c

        !> Fortran interface on routine in C interface of same name.
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

    !> Initialize the runtime.  Calling code should only call this routine after
    !! the Milhoja Grid infrastructure has been initialized.
    !!
    !! @param nThreadTeams        The number of thread teams to create
    !! @param nThreadsPerTeam     The number of threads to create in each team
    !! @param nStreams            The number of streams to create
    !! @param nBytesInMemoryPools The size of memory pools that should be
    !!                            eagerly acquired in all memory spaces.  Note
    !!                            the abnormal integer kind.
    !! @param ierr                The milhoja error code
    subroutine milhoja_runtime_init(nThreadTeams, nThreadsPerTeam, &
                                    nStreams,                      &
                                    nBytesInMemoryPools,           &
                                    ierr)
        integer(MILHOJA_INT),    intent(IN)  :: nThreadTeams
        integer(MILHOJA_INT),    intent(IN)  :: nThreadsPerTeam
        integer(MILHOJA_INT),    intent(IN)  :: nStreams
        integer(MILHOJA_SIZE_T), intent(IN)  :: nBytesInMemoryPools
        integer(MILHOJA_INT),    intent(OUT) :: ierr

        ierr = milhoja_runtime_init_c(nThreadTeams, nThreadsPerTeam, &
                                      nStreams,                      &
                                      nBytesInMemoryPools)
    end subroutine milhoja_runtime_init

    !> Finalize the runtime.  Calling code should finalize the Milhoja 
    !! Grid infrastructure before calling this routine.
    !!
    !! @param ierr    The milhoja error code
    subroutine milhoja_runtime_finalize(ierr)
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ierr = milhoja_runtime_finalize_c()
    end subroutine milhoja_runtime_finalize

    !> Instruct the runtime to use the CPU-only thread team configuration with
    !! the given number of threads to apply the given task function to all
    !! blocks.
    !!
    !! \todo Allow calling code to specify action name for improved logging.
    !! \todo Need to add arguments for specifying the set of blocks.
    !!
    !! @param taskFunction    The task function to execute
    !! @param nThreads        The number of threads to activate in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_executeTasks_Cpu(taskFunction, nThreads, ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: taskFunction
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: taskFunction_Cptr

        taskFunction_Cptr = C_FUNLOC(taskFunction)

        ierr = milhoja_runtime_execute_tasks_cpu_c(taskFunction_Cptr, nThreads)
    end subroutine milhoja_runtime_executeTasks_Cpu

end module milhoja_runtime_mod

