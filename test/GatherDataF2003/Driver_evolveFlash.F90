#include "Flash.h"
#include "constants.h"

program Driver_evolveFlash
    use mpi,                     ONLY : MPI_Wtime, &
                                        MPI_Wtick

    use Grid_interface,          ONLY : Grid_init, &
                                        Grid_finalize, &
                                        Grid_initDomain, &
                                        Grid_getDeltas
    use Orchestration_interface, ONLY : Orchestration_init, &
                                        Orchestration_finalize, &
                                        Orchestration_executeTasks
    use Analysis_interface,      ONLY : Analysis_computeErrors
    use Analysis_data,           ONLY : an_LinfErrors, &
                                        an_meanAbsErrors, &
                                        an_energyFactor
    use Physics_interface,       ONLY : Physics_op1_executeTask1_Tile, &
                                        Physics_op1_executeTask2_Tile, &
                                        Physics_op1_executeTask3_Tile
    use Physics_data,            ONLY : ph_op1_energyFactor  

    implicit none

    ! Have make specify build-time constants for file header
#include "buildInfo.inc"
    integer, parameter :: FOUT = 111

    integer, parameter :: N_THD_TASK1 = 2
    integer, parameter :: N_THD_TASK2 = 2
    integer, parameter :: N_THD_TASK3 = 0

    character(128) :: fname
    real           :: LinfDens
    real           :: LinfEner
    real           :: deltas(1:MDIM)

    double precision    tStart
    double precision    walltime

    call Grid_init()
    call Orchestration_init()

    call Grid_initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, &
                         N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z, NUNKVAR)

    ! This work would be done in init's using a single runtime parameter
    ph_op1_energyFactor = 3.2
    an_energyFactor = ph_op1_energyFactor

    tStart = MPI_Wtime()
    call Orchestration_executeTasks(cpuTask=Physics_op1_executeTask1_Tile, &
                                    nCpuThreads=N_THD_TASK1, &
                                    gpuTask=Physics_op1_executeTask2_Tile, &
                                    nGpuThreads=N_THD_TASK2, &
                                    postGpuTask=Physics_op1_executeTask3_Tile, &
                                    nPostGpuThreads=N_THD_TASK3)
    walltime = MPI_Wtime() - tStart

    ! DEV: Since we are breaking up the high-level operations so that we can
    ! potentially mix tasks from different operations in execution cycles, we
    ! are now doing abnormal things like using physics unit data in Driver.
    !
    ! TODO: Create method in Grid that gets total number of blocks as well as
    !       number of local blocks
    allocate(   an_LinfErrors(1:NUNKVAR, 0:(N_BLOCKS_X*N_BLOCKS_Y*N_BLOCKS_Z)), &
             an_meanAbsErrors(1:NUNKVAR, 0:(N_BLOCKS_X*N_BLOCKS_Y*N_BLOCKS_Z)))

    ! TODO: Here we can get away with running multiple threads running in
    ! parallel and each accessing these arrays without a mutex because we know
    ! that each thread is accessing a unique idx in array.  Does the thread team
    ! need a synchoronization mechanism?!
    an_LinfErrors(:, :)    = 0.0
    an_meanAbsErrors(:, :) = 0.0

    call Orchestration_executeTasks(cpuTask=Analysis_computeErrors, &
                                    nCpuThreads=2)

    LinfDens = MAXVAL(an_LinfErrors(DENS_VAR, :))
    LinfEner = MAXVAL(an_LinfErrors(ENER_VAR, :))
    write(*,*) "Linf           Density = ", LinfDens
    write(*,*) "Linf           Energy  = ", LinfEner

    call Grid_getDeltas(1, deltas)

    write(fname,'(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A)') &
            "gatherDataF2003_", &
            NXB, "_", NYB, "_", NZB, "_", &
            N_BLOCKS_X, "_", N_BLOCKS_Y, "_", N_BLOCKS_Z, &
            ".dat"
    OPEN(unit=FOUT, FILE=fname, STATUS='new', ACTION='write')
    write(FOUT,'(A,A)')       "# Git repository,", PROJECT_GIT_REPO_NAME
    write(FOUT,'(A,A)')       "# Git Commit,", PROJECT_GIT_REPO_VER
!    write(FOUT,'(A,A)')       "# AMReX version,", amrex::Version()
    write(FOUT,'(A,A)')       "# C++ compiler,", CXX_COMPILER
    write(FOUT,'(A,A)')       "# C++ compiler version,", CXX_COMPILER_VERSION
    write(FOUT,'(A,A)')       "# F2003 compiler,", F2003_COMPILER
    write(FOUT,'(A,A)')       "# F2003 compiler version,", F2003_COMPILER_VERSION
    write(FOUT,'(A,A)')       "# Build date,", BUILD_DATETIME
    write(FOUT,'(A,A)')       "# Hostname, ", HOSTNAME
    write(FOUT,'(A,A)')       "# Host information,", MACHINE_INFO
    write(FOUT,'(A,E15.8,A)') "# MPI_Wtick,", MPI_Wtick(), ",sec"
    write(FOUT,'(A)')   "pmode,n_loops,n_thd_task1,n_thd_task2,n_thd_task3," // &
                        "NXB,NYB,NZB,N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z,"      // &
                        "dx,dy,Linf_density,Linf_energy,Walltime_sec"
    write(FOUT,'(A,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,E22.15,A,E22.15,A,E22.15,A,E22.15,A,E22.15)') &
        'Runtime', ",", 1, ",", N_THD_TASK1, ",", N_THD_TASK2, ",", N_THD_TASK3, ",", &
        NXB, ",", NYB, ",", NZB, ",", &
        N_BLOCKS_X, ",", N_BLOCKS_Y, ",", N_BLOCKS_Z, ",", &
        deltas(IAXIS), ",", deltas(JAXIS), ",", &
        LinfDens, ',', LinfEner, ",", walltime
    CLOSE(unit=FOUT)

    deallocate(an_LinfErrors, an_meanAbsErrors)

    call Orchestration_finalize()
    call Grid_finalize()
end program Driver_evolveFlash

