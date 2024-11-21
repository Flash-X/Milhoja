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
#  ifdef RUNTIME_SUPPORT_DATAPACKETS
    public :: milhoja_runtime_setupPipelineForGpuTasks
    public :: milhoja_runtime_pushTileToGpuPipeline
    public :: milhoja_runtime_teardownPipelineForGpuTasks
    public :: milhoja_runtime_setupPipelineForCpuGpuTasks
    public :: milhoja_runtime_pushTileToCpuGpuPipeline
    public :: milhoja_runtime_teardownPipelineForCpuGpuTasks
    public :: milhoja_runtime_setupPipelineForCpuGpuSplitTasks
    public :: milhoja_runtime_pushTileToCpuGpuSplitPipeline
    public :: milhoja_runtime_teardownPipelineForCpuGpuSplitTasks
    public :: milhoja_runtime_setupPipelineForExtGpuTasks
    public :: milhoja_runtime_pushTileToExtGpuPipeline
    public :: milhoja_runtime_teardownPipelineForExtGpuTasks
    public :: milhoja_runtime_setupPipelineForExtCpuGpuSplitTasks
    public :: milhoja_runtime_pushTileToExtCpuGpuSplitPipeline
    public :: milhoja_runtime_teardownPipelineForExtCpuGpuSplitTasks
#  endif
#endif
#ifdef RUNTIME_SUPPORT_EXECUTE
    public :: milhoja_runtime_executeTasks_Cpu
#  ifdef RUNTIME_SUPPORT_DATAPACKETS
    public :: milhoja_runtime_executeTasks_Gpu
    public :: milhoja_runtime_executeTasks_CpuGpu
    public :: milhoja_runtime_executeTasks_CpuGpuSplit
    public :: milhoja_runtime_executeTasks_ExtGpu
    public :: milhoja_runtime_executeTasks_ExtCpuGpuSplit
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
                                                     tileCInfo)  result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),          intent(IN), value :: C_tileWrapperPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
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
        function milhoja_runtime_setup_pipeline_cpugpu_c(C_cpuTaskFunction, &
                                                         C_gpuTaskFunction, &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_packetPrototype) result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_cpuTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_gpuTaskFunction
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_setup_pipeline_cpugpu_c
        function milhoja_runtime_setup_pipeline_cpugpusplit_c(C_cpuTaskFunction, &
                                                         C_gpuTaskFunction, &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_nTilesPerCpuTurn,    &
                                                     C_packetPrototype) result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_cpuTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_gpuTaskFunction
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerCpuTurn
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_setup_pipeline_cpugpusplit_c
        function milhoja_runtime_setup_pipeline_extgpu_c(C_taskFunction,            &
                                                     C_postTaskFunction,    &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_packetPrototype,     &
                                                     C_tilePrototype) result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_taskFunction
            type(C_FUNPTR),       intent(IN), value :: C_postTaskFunction
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_setup_pipeline_extgpu_c
        function milhoja_runtime_setup_pipeline_extcpugpusplit_c(C_cpuTaskFunction,  &
                                                                 C_gpuTaskFunction,  &
                                                                 C_postTaskFunction, &
                                                                 C_nThreads,         &
                                                                 C_nTilesPerPacket,  &
                                                                 C_nTilesPerCpuTurn, &
                                                                 C_packetPrototype,  &
                                                                 C_tilePrototype,    &
                                                                 C_postTilePrototype) result(C_ierr) &
                                                                 bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_cpuTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_gpuTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_postTaskFunction
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            type(C_PTR),          intent(IN), value :: C_postTilePrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerCpuTurn
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_setup_pipeline_extcpugpusplit_c

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
        function milhoja_runtime_teardown_pipeline_cpugpu_c(C_nThreads,            &
                                                         C_nTilesPerPacket)     &
                                                         result(C_ierr) &
                                                         bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_teardown_pipeline_cpugpu_c
        function milhoja_runtime_teardown_pipeline_cpugpusplit_c(C_nThreads,            &
                                                         C_nTilesPerPacket)     &
                                                         result(C_ierr) &
                                                         bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_teardown_pipeline_cpugpusplit_c
        function milhoja_runtime_teardown_pipeline_extgpu_c(C_nThreads,            &
                                                         C_nTilesPerPacket)     &
                                                         result(C_ierr) &
                                                         bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_teardown_pipeline_extgpu_c
        function milhoja_runtime_teardown_pipeline_extcpugpusplit_c(C_nThreads,         &
                                                                    C_nTilesPerPacket)  &
                                                                    result(C_ierr)      &
                                                                    bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_teardown_pipeline_extcpugpusplit_c

        !> Fortran interface for the function in C interface of the same name.
        function milhoja_runtime_push_pipeline_gpu_c(C_packetPrototype,         &
                                                     C_nThreads,                &
                                                     tileCInfo)  result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            type(C_PTR),          intent(IN), value :: tileCInfo
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_push_pipeline_gpu_c
        function milhoja_runtime_push_pipeline_cpugpu_c(C_tilePrototype,        &
                                                        C_packetPrototype,      &
                                                     C_nThreads,                &
                                                     tileCInfo)  result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            type(C_PTR),          intent(IN), value :: tileCInfo
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_push_pipeline_cpugpu_c
        function milhoja_runtime_push_pipeline_cpugpusplit_c(C_tilePrototype,        &
                                                        C_packetPrototype,      &
                                                     C_nThreads,                &
                                                     tileCInfo)  result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            type(C_PTR),          intent(IN), value :: tileCInfo
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_push_pipeline_cpugpusplit_c
        function milhoja_runtime_push_pipeline_extgpu_c(C_packetPrototype,         &
                                                     C_nThreads,                &
                                                     tileCInfo)  result(C_ierr) &
                                                     bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            type(C_PTR),          intent(IN), value :: tileCInfo
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_push_pipeline_extgpu_c
        function milhoja_runtime_push_pipeline_extcpugpusplit_c(C_tilePrototype,           &
                                                                C_packetPrototype,         &
                                                                C_postTilePrototype,       &
                                                                C_nThreads,                &
                                                                tileCInfo)  result(C_ierr) &
                                                                bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            type(C_PTR),          intent(IN), value :: C_postTilePrototype
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            type(C_PTR),          intent(IN), value :: tileCInfo
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_push_pipeline_extcpugpusplit_c

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
        function milhoja_runtime_execute_tasks_cpugpu_c(C_tileTaskFunction, &
                                                     C_pktTaskFunction,     &
                                                     C_nDistributorThreads, &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_packetPrototype,     &
                                                     C_tilePrototype)     &
                                                     result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_tileTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_pktTaskFunction
            integer(MILHOJA_INT), intent(IN), value :: C_nDistributorThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_execute_tasks_cpugpu_c
        function milhoja_runtime_execute_tasks_cpugpusplit_c(C_tileTaskFunction, &
                                                     C_pktTaskFunction,     &
                                                     C_nDistributorThreads, &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_nTilesPerCpuTurn,    &
                                                     C_packetPrototype,     &
                                                     C_tilePrototype)     &
                                                     result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_tileTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_pktTaskFunction
            integer(MILHOJA_INT), intent(IN), value :: C_nDistributorThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerCpuTurn
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_execute_tasks_cpugpusplit_c
        function milhoja_runtime_execute_tasks_extgpu_c(C_taskFunction,     &
                                                     C_postTaskFunction,    &
                                                     C_nDistributorThreads, &
                                                     C_nThreads,            &
                                                     C_nTilesPerPacket,     &
                                                     C_packetPrototype,     &
                                                     C_tilePrototype)     &
                                                     result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_taskFunction
            type(C_FUNPTR),       intent(IN), value :: C_postTaskFunction
            integer(MILHOJA_INT), intent(IN), value :: C_nDistributorThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_execute_tasks_extgpu_c
        function milhoja_runtime_execute_tasks_extcpugpusplit_c(C_cpuTaskFunction,     &
                                                                C_gpuTaskFunction,     &
                                                                C_postTaskFunction,    &
                                                                C_nDistributorThreads, &
                                                                C_nThreads,            &
                                                                C_nTilesPerPacket,     &
                                                                C_nTilesPerCpuTurn,    &
                                                                C_packetPrototype,     &
                                                                C_tilePrototype,       &
                                                                C_postTilePrototype)   &
                                                     result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR, C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),       intent(IN), value :: C_cpuTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_gpuTaskFunction
            type(C_FUNPTR),       intent(IN), value :: C_postTaskFunction
            integer(MILHOJA_INT), intent(IN), value :: C_nDistributorThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerPacket
            integer(MILHOJA_INT), intent(IN), value :: C_nTilesPerCpuTurn
            type(C_PTR),          intent(IN), value :: C_packetPrototype
            type(C_PTR),          intent(IN), value :: C_tilePrototype
            type(C_PTR),          intent(IN), value :: C_postTilePrototype
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_runtime_execute_tasks_extcpugpusplit_c
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

#ifdef RUNTIME_SUPPORT_PUSH
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
    !! @param tileCInfo_Cp    C-pointer to C-compatible tile information,
    !!                        carries identity and properties of the tile
    !!                        and links to actual raw Flash-X real data.
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_pushTileToPipeline(prototype_Cptr, &
                                                nThreads, tileCInfo_Cp, ierr)
        use iso_c_binding, ONLY : C_PTR

        type(C_PTR),                            intent(IN)  :: prototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_push_pipeline_cpu_c(prototype_Cptr, &
                                                   nThreads, &
                                                   tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToPipeline

#  ifdef RUNTIME_SUPPORT_DATAPACKETS
    !> Instruct the runtime to make the GPU-only thread team ready.
    !!
    !! @param taskFunction          The task function to execute
    !! @param packetPrototype_Cptr  C-pointer to a prototype datapacket
    !! @param nThreads              The number of threads to activate in team
    !! @param nTilesPerPacket       The maximum number of tiles in a packet
    !! @param ierr                  The milhoja error code
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
    subroutine milhoja_runtime_setupPipelineForCpuGpuTasks(pktTaskFunction,  &
                                                           tileTaskFunction, &
                                                           nThreads,         &
                                                           nTilesPerPacket,  &
                                                           packetPrototype_Cptr, &
                                                ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: pktTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: tileTaskFunction
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: tileTaskFunction_Cptr
        type(C_FUNPTR) :: pktTaskFunction_Cptr

        tileTaskFunction_Cptr = C_FUNLOC(tileTaskFunction)
        pktTaskFunction_Cptr = C_FUNLOC(pktTaskFunction)

        ierr = milhoja_runtime_setup_pipeline_cpugpu_c(pktTaskFunction_Cptr, &
                                                       tileTaskFunction_Cptr, &
                                                       nThreads, &
                                                       nTilesPerPacket, &
                                                       packetPrototype_Cptr)
    end subroutine milhoja_runtime_setupPipelineForCpuGpuTasks
    subroutine milhoja_runtime_setupPipelineForCpuGpuSplitTasks(pktTaskFunction,  &
                                                           tileTaskFunction, &
                                                           nThreads,         &
                                                           nTilesPerPacket,  &
                                                           nTilesPerCpuTurn,  &
                                                           packetPrototype_Cptr, &
                                                ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: pktTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: tileTaskFunction
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerCpuTurn
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: tileTaskFunction_Cptr
        type(C_FUNPTR) :: pktTaskFunction_Cptr

        tileTaskFunction_Cptr = C_FUNLOC(tileTaskFunction)
        pktTaskFunction_Cptr = C_FUNLOC(pktTaskFunction)

        ierr = milhoja_runtime_setup_pipeline_cpugpusplit_c(pktTaskFunction_Cptr, &
                                                       tileTaskFunction_Cptr, &
                                                       nThreads, &
                                                       nTilesPerPacket, &
                                                       nTilesPerCpuTurn, &
                                                       packetPrototype_Cptr)
    end subroutine milhoja_runtime_setupPipelineForCpuGpuSplitTasks
    subroutine milhoja_runtime_setupPipelineForExtGpuTasks(taskFunction, &
                                                postTaskFunction,     &
                                                nThreads,             &
                                                nTilesPerPacket,      &
                                                packetPrototype_Cptr, &
                                                tilePrototype_Cptr,   &
                                                ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: taskFunction
        procedure(milhoja_runtime_taskFunction)             :: postTaskFunction
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: taskFunction_Cptr
        type(C_FUNPTR) :: postTaskFunction_Cptr

        taskFunction_Cptr = C_FUNLOC(taskFunction)
        postTaskFunction_Cptr = C_FUNLOC(postTaskFunction)

        ierr = milhoja_runtime_setup_pipeline_extgpu_c(taskFunction_Cptr, &
                                                   postTaskFunction_Cptr, &
                                                   nThreads, &
                                                   nTilesPerPacket, &
                                                   packetPrototype_Cptr, &
                                                   tilePrototype_Cptr)
    end subroutine milhoja_runtime_setupPipelineForExtGpuTasks
    subroutine milhoja_runtime_setupPipelineForExtCpuGpuSplitTasks(cpuTaskFunction,        &
                                                                   gpuTaskFunction,        &
                                                                   postTaskFunction,       &
                                                                   nThreads,               &
                                                                   nTilesPerPacket,        &
                                                                   nTilesPerCpuTurn,       &
                                                                   packetPrototype_Cptr,   &
                                                                   tilePrototype_Cptr,     &
                                                                   postTilePrototype_Cptr, &
                                                                   ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: cpuTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: gpuTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: postTaskFunction
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        type(C_PTR),                            intent(IN)  :: postTilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerCpuTurn
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: cpuTaskFunction_Cptr
        type(C_FUNPTR) :: gpuTaskFunction_Cptr
        type(C_FUNPTR) :: postTaskFunction_Cptr

        cpuTaskFunction_Cptr = C_FUNLOC(cpuTaskFunction)
        gpuTaskFunction_Cptr = C_FUNLOC(gpuTaskFunction)
        postTaskFunction_Cptr = C_FUNLOC(postTaskFunction)

        ierr = milhoja_runtime_setup_pipeline_extcpugpusplit_c(cpuTaskFunction_Cptr,  &
                                                               gpuTaskFunction_Cptr,  &
                                                               postTaskFunction_Cptr, &
                                                               nThreads,              &
                                                               nTilesPerPacket,       &
                                                               nTilesPerCpuTurn,      &
                                                               packetPrototype_Cptr,  &
                                                               tilePrototype_Cptr,    &
                                                               postTilePrototype_Cptr)
    end subroutine milhoja_runtime_setupPipelineForExtCpuGpuSplitTasks

    !> Instruct the runtime to tear down the GPU-only thread team pipeline.
    !!
    !! @param nThreads        Number of threads to activate in team (diag)
    !! @param nTilesPerPacket Max number of tiles in a packet (diag)
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_teardownPipelineForGpuTasks(nThreads, nTilesPerPacket,&
                                                           ierr)
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_teardown_pipeline_gpu_c(nThreads, nTilesPerPacket)
    end subroutine milhoja_runtime_teardownPipelineForGpuTasks

    !> Instruct the runtime to tear down the CPUGPU thread team pipeline.
    !!
    !! @param nThreads        Number of threads to activate in team (diag)
    !! @param nTilesPerPacket Max number of tiles in a packet (diag)
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_teardownPipelineForCpuGpuTasks(nThreads, nTilesPerPacket,&
                                                           ierr)
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_teardown_pipeline_cpugpu_c(nThreads, nTilesPerPacket)
    end subroutine milhoja_runtime_teardownPipelineForCpuGpuTasks

    !> Instruct the runtime to tear down the Split CPUGPU thread team pipeline.
    !!
    !! @param nThreads        Number of threads to activate in team (diag)
    !! @param nTilesPerPacket Max number of tiles in a packet (diag)
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_teardownPipelineForCpuGpuSplitTasks(nThreads, nTilesPerPacket,&
                                                           ierr)
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_teardown_pipeline_cpugpusplit_c(nThreads, nTilesPerPacket)
    end subroutine milhoja_runtime_teardownPipelineForCpuGpuSplitTasks

    !> Instruct the runtime to tear down the EXTGPU thread team pipeline.
    !!
    !! @param nThreads        Number of threads to activate in team (diag)
    !! @param nTilesPerPacket Max number of tiles in a packet (diag)
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_teardownPipelineForExtGpuTasks(nThreads, nTilesPerPacket,&
                                                           ierr)
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_teardown_pipeline_extgpu_c(nThreads, nTilesPerPacket)
    end subroutine milhoja_runtime_teardownPipelineForExtGpuTasks
    subroutine milhoja_runtime_teardownPipelineForExtCpuGpuSplitTasks(nThreads, nTilesPerPacket, &
                                                                      ierr)
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_teardown_pipeline_extcpugpusplit_c(nThreads, nTilesPerPacket)
    end subroutine milhoja_runtime_teardownPipelineForExtCpuGpuSplitTasks

    !> Push one tile to the prepared pipeline for task execution.
    !!
    !! @param prototype_Cptr  WRITE THIS
    !! @param nThreads        The number of threads to activate in team
    !! @param tileCInfo_Cp    C-pointer to C-compatible tile information,
    !!                        carries identity and properties of the tile
    !!                        and links to actual raw Flash-X real data.
    !! @param ierr            The milhoja error code
    subroutine milhoja_runtime_pushTileToGpuPipeline(prototype_Cptr, &
                                                nThreads, tileCInfo_Cp, ierr)
        use iso_c_binding, ONLY : C_PTR

        type(C_PTR),                            intent(IN)  :: prototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_push_pipeline_gpu_c(prototype_Cptr, &
                                                   nThreads, &
                                                   tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToGpuPipeline
    subroutine milhoja_runtime_pushTileToCpuGpuPipeline(pktPrototype_Cptr, &
                                                        tilePrototype_Cptr, &
                                                nThreads, tileCInfo_Cp, ierr)
        use iso_c_binding, ONLY : C_PTR

        type(C_PTR),                            intent(IN)  :: pktPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_push_pipeline_cpugpu_c(tilePrototype_Cptr, &
                                                      pktPrototype_Cptr, &
                                                      nThreads, &
                                                      tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToCpuGpuPipeline
    subroutine milhoja_runtime_pushTileToCpuGpuSplitPipeline(pktPrototype_Cptr, &
                                                        tilePrototype_Cptr, &
                                                nThreads, tileCInfo_Cp, ierr)
        use iso_c_binding, ONLY : C_PTR

        type(C_PTR),                            intent(IN)  :: pktPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_push_pipeline_cpugpusplit_c(tilePrototype_Cptr, &
                                                      pktPrototype_Cptr, &
                                                      nThreads, &
                                                      tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToCpuGpuSplitPipeline
    subroutine milhoja_runtime_pushTileToExtGpuPipeline(prototype_Cptr, &
                                                nThreads, tileCInfo_Cp, ierr)
        use iso_c_binding, ONLY : C_PTR

        type(C_PTR),                            intent(IN)  :: prototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        ierr = milhoja_runtime_push_pipeline_extgpu_c(prototype_Cptr, &
                                                   nThreads, &
                                                   tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToExtGpuPipeline
    subroutine milhoja_runtime_pushTileToExtCpuGpuSplitPipeline(tilePrototype_Cptr, &
                                                                pktPrototype_Cptr, &
                                                                postTilePrototype_Cptr, &
                                                                nThreads, &
                                                                tileCInfo_Cp, &
                                                                ierr)
        use iso_c_binding, ONLY : C_PTR

        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        type(C_PTR),                            intent(IN)  :: pktPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: postTilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        type(C_PTR),                            intent(IN)  :: tileCInfo_Cp
        integer(MILHOJA_INT),                   intent(OUT) :: ierr
        ierr = milhoja_runtime_push_pipeline_extcpugpusplit_c(tilePrototype_Cptr, &
                                                              pktPrototype_Cptr, &
                                                              postTilePrototype_Cptr, &
                                                              nThreads, &
                                                              tileCInfo_Cp)
    end subroutine milhoja_runtime_pushTileToExtCpuGpuSplitPipeline
#  endif
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

    !> Instruct the runtime to use the CPU/GPU thread team configuration
    !! with the given number of threads to apply the given task function to all
    !! blocks in packet form ("on the GPU") and then the "Post" task function
    !! in direct (tile-wrapped) form ("on the CPU").
    !! blocks.
    !!
    !! \todo Allow calling code to specify action name for improved logging.
    !! \todo Should add arguments for specifying the set of blocks.
    !!
    !! @param tileTaskFunction      The task function to execute "on the CPU"
    !! @param pktTaskFunction       The packet task function to execute "on the GPU"
    !! @param nDistributorThreads   The number of distributor threads to use
    !! @param nThreads              The number of threads to activate in team
    !! @param nTilesPerPacket       The maximum number of tiles allowed in each
    !!                              packet
    !! @param packetPrototype_Cptr  Pointer to a prototype data packet to be
    !!                              used to create new packets.
    !! @param tilePrototype_Cptr    Pointer to a prototype tile wrapper to be
    !!                              used to enqueue tiles.
    !! @param ierr                  The milhoja error code
    subroutine milhoja_runtime_executeTasks_CpuGpu(tileTaskFunction,  &
                                                pktTaskFunction,      &
                                                nDistributorThreads,  &
                                                nThreads,             &
                                                nTilesPerPacket,      &
                                                packetPrototype_Cptr, &
                                                tilePrototype_Cptr,   &
                                                ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_PTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: tileTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: pktTaskFunction
        integer(MILHOJA_INT),                   intent(IN)  :: nDistributorThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: tileTaskFunction_Cptr
        type(C_FUNPTR) :: pktTaskFunction_Cptr

        tileTaskFunction_Cptr = C_FUNLOC(tileTaskFunction)
        pktTaskFunction_Cptr = C_FUNLOC(pktTaskFunction)

        ierr = milhoja_runtime_execute_tasks_cpugpu_c(tileTaskFunction_Cptr, &
                                                   pktTaskFunction_Cptr, &
                                                   nDistributorThreads, &
                                                   nThreads, &
                                                   nTilesPerPacket, &
                                                   packetPrototype_Cptr, &
                                                   tilePrototype_Cptr)
    end subroutine milhoja_runtime_executeTasks_CpuGpu

    !> Instruct the runtime to use the Split CPU/GPU thread team configuration
    !! with the given number of threads to apply the given packet task
    !! function to some of the blocks in packet form ("on the GPU") and
    !! and the tile task function in direct (tile-wrapped) form ("on the CPU")
    !! to the other blocks, in an alternating fashion.
    !!
    !! \todo Allow calling code to specify action name for improved logging.
    !! \todo Should add arguments for specifying the set of blocks.
    !!
    !! @param tileTaskFunction      The task function to execute "on the CPU"
    !! @param pktTaskFunction       The packet task function to execute "on the GPU"
    !! @param nDistributorThreads   The number of distributor threads to use
    !! @param nThreads              The number of threads to activate in team
    !! @param nTilesPerPacket       The maximum number of tiles allowed in each
    !!                              packet
    !! @param nTilesPerCpuTurn      A CpuGpuSplit distributor thread will hand the first
    !!                              `nTilesPerCpuTurn` tiles from the grid iterator to
    !!                              the cpuTaskFunction (as separate tasks), then build
    !!                              a data packet from the next `nTilesPerPacket` tiles
    !!                              and - when full - hand it to the gpuTaskFunction as
    !!                              one task; then repeat; until the iterator is exhausted.
    !! @param packetPrototype_Cptr  Pointer to a prototype data packet to be
    !!                              used to create new packets.
    !! @param tilePrototype_Cptr    Pointer to a prototype tile wrapper to be
    !!                              used to enqueue tiles.
    !! @param ierr                  The milhoja error code
    subroutine milhoja_runtime_executeTasks_CpuGpuSplit(tileTaskFunction,  &
                                                pktTaskFunction,      &
                                                nDistributorThreads,  &
                                                nThreads,             &
                                                nTilesPerPacket,      &
                                                nTilesPerCpuTurn,     &
                                                packetPrototype_Cptr, &
                                                tilePrototype_Cptr,   &
                                                ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_PTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: tileTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: pktTaskFunction
        integer(MILHOJA_INT),                   intent(IN)  :: nDistributorThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerCpuTurn
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: tileTaskFunction_Cptr
        type(C_FUNPTR) :: pktTaskFunction_Cptr

        tileTaskFunction_Cptr = C_FUNLOC(tileTaskFunction)
        pktTaskFunction_Cptr = C_FUNLOC(pktTaskFunction)

        ierr = milhoja_runtime_execute_tasks_cpugpusplit_c(tileTaskFunction_Cptr, &
                                                   pktTaskFunction_Cptr, &
                                                   nDistributorThreads, &
                                                   nThreads, &
                                                   nTilesPerPacket, &
                                                   nTilesPerCpuTurn, &
                                                   packetPrototype_Cptr, &
                                                   tilePrototype_Cptr)
    end subroutine milhoja_runtime_executeTasks_CpuGpuSplit

    !> Instruct the runtime to use the GPU/post-GPU thread team configuration
    !! with the given number of threads to apply the given task function to all
    !! blocks in packet form ("on the GPU") and then the "Post" task function
    !! in direct (tile-wrapped) form ("on the CPU").
    !! blocks.
    !!
    !! \todo Allow calling code to specify action name for improved logging.
    !! \todo Should add arguments for specifying the set of blocks.
    !!
    !! @param taskFunction          The task function to execute "on the GPU"
    !! @param postTaskFunction      The "Post" task function to execute "on the CPU"
    !! @param nDistributorThreads   The number of distributor threads to use
    !! @param nThreads              The number of threads to activate in team
    !! @param nTilesPerPacket       The maximum number of tiles allowed in each
    !!                              packet
    !! @param packetPrototype_Cptr  Pointer to a prototype data packet to be
    !!                              used to create new packets.
    !! @param tilePrototype_Cptr    Pointer to a prototype tile wrapper to be
    !!                              used to enqueue tiles.
    !! @param ierr                  The milhoja error code
    subroutine milhoja_runtime_executeTasks_ExtGpu(taskFunction,      &
                                                postTaskFunction,     &
                                                nDistributorThreads,  &
                                                nThreads,             &
                                                nTilesPerPacket,      &
                                                packetPrototype_Cptr, &
                                                tilePrototype_Cptr,   &
                                                ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_PTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: taskFunction
        procedure(milhoja_runtime_taskFunction)             :: postTaskFunction
        integer(MILHOJA_INT),                   intent(IN)  :: nDistributorThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: taskFunction_Cptr
        type(C_FUNPTR) :: postTaskFunction_Cptr

        taskFunction_Cptr = C_FUNLOC(taskFunction)
        postTaskFunction_Cptr = C_FUNLOC(postTaskFunction)

        ierr = milhoja_runtime_execute_tasks_extgpu_c(taskFunction_Cptr, &
                                                   postTaskFunction_Cptr, &
                                                   nDistributorThreads, &
                                                   nThreads, &
                                                   nTilesPerPacket, &
                                                   packetPrototype_Cptr, &
                                                   tilePrototype_Cptr)
    end subroutine milhoja_runtime_executeTasks_ExtGpu


    subroutine milhoja_runtime_executeTasks_ExtCpuGpuSplit(tileTaskFunction,       &
                                                           pktTaskFunction,        &
                                                           postTaskFunction,       &
                                                           nDistributorThreads,    &
                                                           nThreads,               &
                                                           nTilesPerPacket,        &
                                                           nTilesPerCpuTurn,       &
                                                           packetPrototype_Cptr,   &
                                                           tilePrototype_Cptr,     &
                                                           postTilePrototype_Cptr, &
                                                           ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_PTR, &
                                  C_FUNLOC

        procedure(milhoja_runtime_taskFunction)             :: tileTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: pktTaskFunction
        procedure(milhoja_runtime_taskFunction)             :: postTaskFunction
        integer(MILHOJA_INT),                   intent(IN)  :: nDistributorThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nThreads
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerPacket
        integer(MILHOJA_INT),                   intent(IN)  :: nTilesPerCpuTurn
        type(C_PTR),                            intent(IN)  :: packetPrototype_Cptr
        type(C_PTR),                            intent(IN)  :: tilePrototype_Cptr
        type(C_PTR),                            intent(IN)  :: postTilePrototype_Cptr
        integer(MILHOJA_INT),                   intent(OUT) :: ierr

        type(C_FUNPTR) :: tileTaskFunction_Cptr
        type(C_FUNPTR) :: pktTaskFunction_Cptr
        type(C_FUNPTR) :: postTaskFunction_Cptr

        tileTaskFunction_Cptr = C_FUNLOC(tileTaskFunction)
        pktTaskFunction_Cptr = C_FUNLOC(pktTaskFunction)
        postTaskFunction_Cptr = C_FUNLOC(postTaskFunction)

        ierr = milhoja_runtime_execute_tasks_extcpugpusplit_c(tileTaskFunction_Cptr, &
                                                              pktTaskFunction_Cptr,  &
                                                              postTaskFunction_Cptr, &
                                                              nDistributorThreads,   &
                                                              nThreads,              &
                                                              nTilesPerPacket,       &
                                                              nTilesPerCpuTurn,      &
                                                              packetPrototype_Cptr,  &
                                                              tilePrototype_Cptr,    &
                                                              postTilePrototype_Cptr)
    end subroutine milhoja_runtime_executeTasks_ExtCpuGpuSplit
#  endif
#endif

end module milhoja_runtime_mod

