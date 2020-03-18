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
                                        an_meanAbsErrors

    implicit none

    call Grid_init()
    call Orchestration_init()

    call Grid_initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, &
                         N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z, NUNKVAR)

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
    call Orchestration_executeTasks(Analysis_computeErrors, 2)
    write(*,*) "Linf           Density = ", MAXVAL(an_LinfErrors(DENS_VAR, :))
    write(*,*) "Linf           Energy  = ", MAXVAL(an_LinfErrors(ENER_VAR, :))

    deallocate(an_LinfErrors, an_meanAbsErrors)

    call Orchestration_finalize()
    call Grid_finalize()

end program Driver_evolveFlash

