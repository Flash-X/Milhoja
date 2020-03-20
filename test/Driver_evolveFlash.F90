#include "Flash.h"
#include "constants.h"

program Driver_evolveFlash
    use Grid_interface,          ONLY : Grid_init, &
                                        Grid_finalize, &
                                        Grid_initDomain
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

    call Grid_init()
    call Orchestration_init()

    call Grid_initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, &
                         N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z, NUNKVAR)

    ! This work would be done in init's using a single runtime parameter
    ph_op1_energyFactor = 3.2
    an_energyFactor = ph_op1_energyFactor

    call Orchestration_executeTasks(cpuTask=Physics_op1_executeTask1_Tile, &
                                    nCpuThreads=2, &
                                    gpuTask=Physics_op1_executeTask2_Tile, &
                                    nGpuThreads=2, &
                                    postGpuTask=Physics_op1_executeTask3_Tile, &
                                    nPostGpuThreads=0)

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
    write(*,*) "Linf           Density = ", MAXVAL(an_LinfErrors(DENS_VAR, :))
    write(*,*) "Linf           Energy  = ", MAXVAL(an_LinfErrors(ENER_VAR, :))

    deallocate(an_LinfErrors, an_meanAbsErrors)

    call Orchestration_finalize()
    call Grid_finalize()

end program Driver_evolveFlash

