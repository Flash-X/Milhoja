#include "Milhoja.h"

!> A module in the Milhoja Fortran/C++ interoperability layer that provides
!! calling code with a high-level Fortan interface for interacting with the runtime.
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
    public :: milhoja_runtime_reset
#ifdef RUNTIME_SUPPORT_PUSH
    public :: milhoja_runtime_setupPipelineForCpuTasks
    public :: milhoja_runtime_pushTileToPipeline
    public :: milhoja_runtime_teardownPipelineForCpuTasks
    public :: milhoja_runtime_setupPipelineForGpuTasks
    public :: milhoja_runtime_pushTileToGpuPipeline
    public :: milhoja_runtime_teardownPipelineForGpuTasks
#endif
#ifdef RUNTIME_SUPPORT_EXECUTE
    public :: milhoja_runtime_executeTasks_Cpu
#  ifdef RUNTIME_SUPPORT_DATAPACKETS
    public :: milhoja_runtime_executeTasks_Gpu
#  endif
#endif

    !!!!!----- FORTRAN INTERFACES TO MILHOJA FUNCTION POINTERS
    abstract interface
        !> Fortran interface of the runtime's task function.
        !!
        !! C_threadId - unique zero-based index of runtime thread calling this
        !!              routine
        !! C_dataItemPtr - C pointer to Grid DataItem to which the task
        !!                 function should be applied
        subroutine milhoja_runtime_taskFunction(C_threadId, C_dataItemPtr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_threadId
            type(C_PTR),          intent(IN), value :: C_dataItemPtr
        end subroutine milhoja_runtime_taskFunction
    end interface

    !!!!!----- INTERFACES TO C-LINKAGE C++ FUNCTIONS
    ! The C-to-Fortran interoperability layer
    interface
        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_init_c(C_nThreadTeams, C_nThreadsPerTeam, &
                                        C_nStreams,                        &
                                        C_nBytesInCpuMemoryPool,           &
                                        C_nBytesInGpuMemoryPools) result(C_ierr) &
                                        bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_SIZE_T
            implicit none
            integer(MILHOJA_INT),    intent(IN), value :: C_nThreadTeams
            integer(MILHOJA_INT),    intent(IN), value :: C_nThreadsPerTeam
            integer(MILHOJA_INT),    intent(IN), value :: C_nStreams
            integer(MILHOJA_SIZE_T), intent(IN), value :: C_nBytesInCpuMemoryPool
            integer(MILHOJA_SIZE_T), intent(IN), value :: C_nBytesInGpuMemoryPools
            integer(MILHOJA_INT)                       :: C_ierr
        end function milhoja_runtime_init_c

        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_finalize_c() result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT) :: C_ierr
        end function milhoja_runtime_finalize_c

        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_reset_c() result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT) :: C_ierr
        end function milhoja_runtime_reset_c

#ifdef RUNTIME_SUPPORT_EXECUTE
        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_execute_tasks_cpu_c(C_taskFunction,            &
                                                     C_tileWrapperPrototype,    &
                                                     C_nThreads) result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_taskFunction
            type(C_PTR),          intent(IN), value :: C_tileWrapperPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_execute_tasks_cpu_c
#endif

        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_setup_pipeline_cpu_c(C_taskFunction,            &
                                                     C_nThreads) result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_taskFunction
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_setup_pipeline_cpu_c

        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_teardown_pipeline_cpu_c(C_nThreads) result(C_ierr) &
                                                     bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_teardown_pipeline_cpu_c

        !> Fortran interface on routine in C interface of same name.
        function milhoja_runtime_push_pipeline_cpu_c(C_tileWrapperPrototype,    &
                                                     C_nThreads,                &
                                                     tileCINfo)  result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
!!$            use Milhoja_tileCInfo_mod, ONLY: Milhoja_tileCInfo_t
            implicit none
            type(C_PTR),          intent(IN), value :: C_tileWrapperPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
!!$            type(Milhoja_tileCInfo_t), intent(IN)   :: tileCInfo
            type(C_PTR),          intent(IN), value :: tileCInfo
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_push_pipeline_cpu_c

#ifdef RUNTIME_SUPPORT_DATAPACKETS
        !> Fortran interface for the function in C interface of the same name.
        function milhoja_runtime_setup_pipeline_gpu_c(C_taskFunction,            &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_packetPrototype) result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_taskFunction
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_setup_pipeline_gpu_c

        !> Fortran interface for the function in C interface of the same name.
        function milhoja_runtime_teardown_pipeline_gpu_c(C_nThreads,            &
                                                         C_nTilesPerPacket)     &
                                                         result(C_ierr) &
                                                         bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_teardown_pipeline_gpu_c

        !> Fortran interface for the function in C interface of the same name.
        function milhoja_runtime_push_pipeline_gpu_c(C_packetPrototype,         &
                                                     C_nThreads,                &
                                                     tileCINfo)  result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
!!$            use Milhoja_tileCInfo_mod, ONLY: Milhoja_tileCInfo_t
            implicit none
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
!!$            type(Milhoja_tileCInfo_t), intent(IN)   :: tileCInfo
            type(C_PTR),          intent(IN), value :: tileCInfo
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_push_pipeline_gpu_c

#  ifdef RUNTIME_SUPPORT_EXECUTE
        !> Fortran interface for the function in C interface of the same name.
        function milhoja_runtime_execute_tasks_gpu_c(C_taskFunction,        &
                                                     C_nDistributorThreads, &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_packetPrototype)     &
                                                     result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_taskFunction
            integer(MILHOJA_INT), intent(IN), value :: C_nDistributorThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_execute_tasks_gpu_c
#  endif
#endif
    end interface

contains

    !> Initialize the runtime.  Calling code should only call this routine after
    !! the Milhoja Grid infrastructure has been initialized.
    !!
    !! @param nThreadTeams           Number of thread teams to create
    !! @param nThreadsPerTeam        Number of threads to create in each team
    !! @param nStreams               Number of streams to create
    !! @param nBytesInCpuMemoryPool  Number of bytes to allocate in CPU memory pool.
    !!                               Note the abnormal integer kind.
    !! @param nBytesInGpuMemoryPools Number of bytes to allocate in memory pools
    !!                               associated with GPU (e.g., pinned & GPU).
    !!                               Note the abnormal integer kind.
    !! @param ierr                   Milhoja error code
    subroutine milhoja_runtime_init(nThreadTeams, nThreadsPerTeam, &
                                    nStreams,                      &
                                    nBytesInCpuMemoryPool,         &
                                    nBytesInGpuMemoryPools,        &
                                    ierr)
        integer(MILHOJA_INT),    intent(IN)  :: nThreadTeams
        integer(MILHOJA_INT),    intent(IN)  :: nThreadsPerTeam
        integer(MILHOJA_INT),    intent(IN)  :: nStreams
        integer(MILHOJA_SIZE_T), intent(IN)  :: nBytesInCpuMemoryPool
        integer(MILHOJA_SIZE_T), intent(IN)  :: nBytesInGpuMemoryPools
        integer(MILHOJA_INT),    intent(OUT) :: ierr

        ierr = milhoja_runtime_init_c(nThreadTeams, nThreadsPerTeam, &
                                      nStreams,                      &
                                      nBytesInCpuMemoryPool,         &
                                      nBytesInGpuMemoryPools)
    end subroutine milhoja_runtime_init

    !> Finalize the runtime.  Calling code should call this routine before
    !! finalizing the Milhoja Grid infrastructure.
    !!
    !! @param ierr    The milhoja error code
    subroutine milhoja_runtime_finalize(ierr)
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ierr = milhoja_runtime_finalize_c()
    end subroutine milhoja_runtime_finalize

    !> Reset the runtime backend.  This is a temporary workaround required
    !! since the present memory manager is too simple.
    !!
    !! @todo Remove once a proper memory manager is implemented.
    !!
    !! @param ierr    The milhoja error code
    subroutine milhoja_runtime_reset(ierr)
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ierr = milhoja_runtime_reset_c()
    end subroutine milhoja_runtime_reset

    !> Instruct the runtime to make the CPU-only thread team ready.
    !!
    !! @param taskFunction    The task function to execute
    !! @param nThreads        The number of threads to activate in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_setupPipelineForCpuTasks(taskFunction, &
                                                nThreads, ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: taskFunction
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: taskFunction_Cptr

        taskFunction_Cptr = C_FUNLOC(taskFunction)

        ierr = milhoja_runtime_setup_pipeline_cpu_c(taskFunction_Cptr, &
                                                   nThreads)
    end subroutine milhoja_runtime_setupPipelineForCpuTasks

    !> Instruct the runtime to tear down the CPU-only thread team pipeline.
    !!
    !! @param nThreads        The number of threads that should be activated in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_teardownPipelineForCpuTasks(nThreads, ierr)

        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_teardown_pipeline_cpu_c(nThreads)
    end subroutine milhoja_runtime_teardownPipelineForCpuTasks

    !> Push one tile to the prepared pipeline for task execution.
    !!
    !! @param prototype_Cptr  WRITE THIS
    !! @param nThreads        The number of threads to activate in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_pushTileToPipeline(prototype_Cptr, &
                                                nThreads, tileCInfo_Cp, ierr)
        use iso_c_binding, ONLY : C_PTR
!!$        use Milhoja_tileCInfo_mod, ONLY: Milhoja_tileCInfo_t

        type(C_PTR),                            intent(IN)  :: prototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
!!$        type(Milhoja_tileCInfo_t),              intent(IN)  :: tileCInfo
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_push_pipeline_cpu_c(prototype_Cptr, &
                                                   nThreads, &
                                                   tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToPipeline

#ifdef RUNTIME_SUPPORT_PUSH
    !> Instruct the runtime to make the GPU-only thread team ready.
    !!
    !! @param taskFunction    The task function to execute
    !! @param prototype_Cptr  WRITE THIS
    !! @param nThreads        The number of threads to activate in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_setupPipelineForGpuTasks(taskFunction, &
                                                nThreads,             &
                                                nTilesPerPacket,      &
                                                packetPrototype_Cptr, &
                                                ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: taskFunction
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: taskFunction_Cptr

        taskFunction_Cptr = C_FUNLOC(taskFunction)

        ierr = milhoja_runtime_setup_pipeline_gpu_c(taskFunction_Cptr, &
                                                   nThreads, &
                                                   nTilesPerPacket, &
                                                   packetPrototype_Cptr)
    end subroutine milhoja_runtime_setupPipelineForGpuTasks

    !> Instruct the runtime to tear down the GPU-only thread team pipeline.
    !!
    !! @param nThreads        The number of threads to activate in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_teardownPipelineForGpuTasks(nThreads, nTilesPerPacket,&
                                                           ierr)
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_teardown_pipeline_gpu_c(nThreads, nTilesPerPacket)
    end subroutine milhoja_runtime_teardownPipelineForGpuTasks

    !> Push one tile to the prepared pipeline for task execution.
    !!
    !! @param prototype_Cptr  WRITE THIS
    !! @param nThreads        The number of threads to activate in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_pushTileToGpuPipeline(prototype_Cptr, &
                                                nThreads, tileCInfo_Cp, ierr)
        use iso_c_binding, ONLY : C_PTR
!!$        use Milhoja_tileCInfo_mod, ONLY: Milhoja_tileCInfo_t

        type(C_PTR),                            intent(IN)  :: prototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
!!$        type(Milhoja_tileCInfo_t),              intent(IN)  :: tileCInfo
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_push_pipeline_gpu_c(prototype_Cptr, &
                                                   nThreads, &
                                                   tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToGpuPipeline
#endif

#ifdef RUNTIME_SUPPORT_EXECUTE
    !> Instruct the runtime to use the CPU-only thread team configuration with
    !! the given number of threads to apply the given task function to all
    !! blocks.
    !!
    !! \todo Allow calling code to specify action name for improved logging.
    !! \todo Need to add arguments for specifying the set of blocks.
    !!
    !! @param taskFunction    The task function to execute
    !! @param prototype_Cptr  WRITE THIS
    !! @param nThreads        The number of threads to activate in team
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_executeTasks_Cpu(taskFunction, &
                                                prototype_Cptr, &
                                                nThreads, ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: taskFunction
        type(C_PTR),                            intent(IN)  :: prototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: taskFunction_Cptr

        taskFunction_Cptr = C_FUNLOC(taskFunction)

        ierr = milhoja_runtime_execute_tasks_cpu_c(taskFunction_Cptr, &
                                                   prototype_Cptr, &
                                                   nThreads)
    end subroutine milhoja_runtime_executeTasks_Cpu

#  ifdef RUNTIME_SUPPORT_DATAPACKETS
    !> Instruct the runtime to use the GPU-only thread team configuration with
    !! the given number of threads to apply the given task function to all
    !! blocks.
    !!
    !! \todo Allow calling code to specify action name for improved logging.
    !! \todo Need to add arguments for specifying the set of blocks.
    !!
    !! @param taskFunction          The task function to execute
    !! @param nDistributorThreads   The number of distributor threads to use
    !! @param nThreads              The number of threads to activate in team
    !! @param nTilesPerPacket       The maximum number of tiles allowed in each
    !!                              packet
    !! @param packetPrototype_Cptr  Pointer to a prototype data packet to be
    !!                              used to create new packets.
    !! @param ierr                  The milhoja error code
    subroutine milhoja_runtime_executeTasks_Gpu(taskFunction,         &
                                                nDistributorThreads,  &
                                                nThreads,             &
                                                nTilesPerPacket,      &
                                                packetPrototype_Cptr, &
                                                ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_PTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: taskFunction
        integer(MILHOJA_INT),                   intent(IN)  :: nDistributorThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: taskFunction_Cptr

        taskFunction_Cptr = C_FUNLOC(taskFunction)

        ierr = milhoja_runtime_execute_tasks_gpu_c(taskFunction_Cptr, &
                                                   nDistributorThreads, &
                                                   nThreads, &
                                                   nTilesPerPacket, &
                                                   packetPrototype_Cptr)
    end subroutine milhoja_runtime_executeTasks_Gpu
#  endif
#endif

end module milhoja_runtime_mod

