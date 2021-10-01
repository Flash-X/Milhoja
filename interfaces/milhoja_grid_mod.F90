!> A module in the Milhoja Fortran/C++ interoperability layer that provides
!! calling code with a high-level Fortran interface for interacting with
!! the the Milhoja grid infrastructure.
!!
!! @todo Build out full interface including iterator, AMR functionality,
!!       and flux correction.
!! @todo constants.h/Flash.h should be replaced by a milhoja-specific
!!       file with a unique name that doesn't clash with Flash-X's files.
!! @todo Add in information about indexing or point to information in another
!!       file's documentation.

#include "milhoja_interface_error_codes.h"

#include "constants.h"

module milhoja_grid_mod
    use milhoja_types_mod, ONLY : MILHOJA_INT, &
                                  MILHOJA_REAL

    implicit none
    private

    !!!!!----- PUBLIC INTERFACE
    public :: milhoja_grid_init
    public :: milhoja_grid_finalize
    public :: milhoja_grid_initDomain
    public :: milhoja_grid_getMaxFinestLevel
    public :: milhoja_grid_getCurrentFinestLevel
    public :: milhoja_grid_getDomainBoundBox
    public :: milhoja_grid_getDeltas
    public :: milhoja_grid_writePlotfile

    !!!!!----- FORTRAN INTERFACES TO MILHOJA FUNCTION POINTERS
    abstract interface
        !> Fortran interface of the callback registered with the grid
        !! infrastructure for the purposes of estimating error in all given
        !! cells due to insufficient mesh refinement.
        !!
        !! @todo If this were AMReX, we should be using the AMReX kinds
        !!      for integer and real.  Should this be generic?  If so,
        !!      how do we make certain that the AMReX kinds match these
        !!      C-compatible kinds?
        subroutine milhoja_errorEstimateCallBack(level, tags, time, &
                                                 tagval, clearval) bind(c)
            use iso_c_binding,     ONLY : C_CHAR, C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_REAL
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: level
            type(C_PTR),          intent(IN), value :: tags 
            real(MILHOJA_REAL),   intent(IN), value :: time
            character(C_CHAR),    intent(IN), value :: tagval
            character(C_CHAR),    intent(IN), value :: clearval
        end subroutine milhoja_errorEstimateCallback
    end interface

    !!!!!----- INTERFACES TO C-LINKAGE C++ FUNCTIONS
    ! The C-to-Fortran interoperability layer
    interface
        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_init_C(C_globalCommF,                      &
                                     C_xMin, C_xMax,                     &
                                     C_yMin, C_yMax,                     &
                                     C_zMin, C_zMax,                     &
                                     C_nxb, C_nyb, C_nzb,                &
                                     C_nBlocksX, C_nBlocksY, C_nBlocksZ, &
                                     C_lRefineMax,                       &
                                     C_nGuard, C_nCcVars,                &
                                     C_initBlock, C_errorEst) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_REAL
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_globalCommF
            real(MILHOJA_REAL),   intent(IN), value :: C_xMin, C_xMax
            real(MILHOJA_REAL),   intent(IN), value :: C_yMin, C_yMax
            real(MILHOJA_REAL),   intent(IN), value :: C_zMin, C_zMax
            integer(MILHOJA_INT), intent(IN), value :: C_nxb, C_nyb, C_nzb
            integer(MILHOJA_INT), intent(IN), value :: C_nBlocksX, C_nBlocksY, C_nBlocksZ
            integer(MILHOJA_INT), intent(IN), value :: C_lRefineMax
            integer(MILHOJA_INT), intent(IN), value :: C_nGuard
            integer(MILHOJA_INT), intent(IN), value :: C_nCcVars
            type(C_FUNPTR),       intent(IN), value :: C_initBlock
            type(C_FUNPTR),       intent(IN), value :: C_errorEst
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_grid_init_C

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_finalize_C() result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT) :: C_ierr
        end function milhoja_grid_finalize_C

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_init_domain_C(C_nDistributorThreads,         &
                                            C_nTeamThreads) result(C_ierr) &
                                            bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_nDistributorThreads
            integer(MILHOJA_INT), intent(IN), value :: C_nTeamThreads
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_grid_init_domain_C

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_domain_bound_box_C(C_lo, C_hi) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),         intent(IN), value :: C_lo
            type(C_PTR),         intent(IN), value :: C_hi
            integer(MILHOJA_INT)                   :: C_ierr
        end function milhoja_grid_domain_bound_box_C

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_max_finest_level_C(C_level) result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(OUT) :: C_level
            integer(MILHOJA_INT)              :: C_ierr
        end function milhoja_grid_max_finest_level_C

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_current_finest_level_C(C_level) result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(OUT) :: C_level
            integer(MILHOJA_INT)              :: C_ierr
        end function milhoja_grid_current_finest_level_C

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_deltas_C(C_level, C_deltas) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_level
            type(C_PTR),          intent(IN), value :: C_deltas
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_grid_deltas_C

        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_grid_write_plotfile_C(C_step) result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_step
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_grid_write_plotfile_C
    end interface

contains

    !> Initialize the grid infrastructure.  This assumes that MPI has already
    !! been initialized by the calling code.
    !!
    !! @todo Does this unit or the runtime need to be initialized
    !!       first?  If so, document here and in runtime.
    !! @todo Should confirmation of correct types be logged?
    !! @todo This should not make the decision to print type info to stdout
    !!       It should be logged internally or calling code should pass in
    !!       a file unit.  Check with xSDK.
    !!
    !! @param F_globalCommF  The Fortran version of the MPI communicator that
    !!                       Milhoja should use
    !! @param F_xMin         Define the physical domain in X as [F_xMin, F_xMax]
    !! @param F_xMax         Define the physical domain in X as [F_xMin, F_xMax]
    !! @param F_yMin         Define the physical domain in Y as [F_yMin, F_yMax]
    !! @param F_yMax         Define the physical domain in Y as [F_yMin, F_yMax]
    !! @param F_zMin         Define the physical domain in Z as [F_zMin, F_zMax]
    !! @param F_zMax         Define the physical domain in Z as [F_zMin, F_zMax]
    !! @param F_nxb          The number of cells along X in each block in the
    !!                       domain decomposition
    !! @param F_nyb          The number of cells along Y in each block in the
    !!                       domain decomposition
    !! @param F_nzb          The number of cells along Z in each block in the
    !!                       domain decomposition
    !! @param F_nBlocksX     The number of blocks along X in the domain decomposition
    !! @param F_nBlocksY     The number of blocks along Y in the domain decomposition
    !! @param F_nBlocksZ     The number of blocks along Z in the domain decomposition
    !! @param F_lRefineMax   The 1-based index of the finest refinement level
    !!                       permitted at any time during the simulation
    !! @param F_nGuard       The number of guardcells
    !! @param F_nCcVars      The number of physical variables in the solution
    !! @param F_initBlock    Procedure to use to compute and store the initial
    !!                       conditions
    !! @param F_errorEst     Procedure that is used to assess if a block should
    !!                       be refined, derefined, or stay at the same
    !!                       refinement
    !! @param F_ierr         The milhoja error code
    subroutine milhoja_grid_init(F_globalCommF,                      &
                                 F_xMin, F_xMax,                     &
                                 F_yMin, F_yMax,                     &
                                 F_zMin, F_zMax,                     &
                                 F_nxb, F_nyb, F_nzb,                &
                                 F_nBlocksX, F_nBlocksY, F_nBlocksZ, &
                                 F_lRefineMax,                       &
                                 F_nGuard, F_nCcVars,                &
                                 F_initBlock, F_errorEst,            &
                                 F_ierr)
        use mpi
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_FUNLOC


        use milhoja_types_mod,   ONLY : milhoja_types_confirmMatchingTypes, &
                                        milhoja_types_printTypesInformation
        use milhoja_runtime_mod, ONLY : milhoja_runtime_taskFunction

        integer, parameter :: STDOUT = 6

        integer, intent(IN)                      :: F_globalCommF
        real,    intent(IN)                      :: F_xMin, F_xMax
        real,    intent(IN)                      :: F_yMin, F_yMax
        real,    intent(IN)                      :: F_zMin, F_zMax
        integer, intent(IN)                      :: F_nxb, F_nyb, F_nzb
        integer, intent(IN)                      :: F_nBlocksX, F_nBlocksY, F_nBlocksZ
        integer, intent(IN)                      :: F_lRefineMax
        integer, intent(IN)                      :: F_nGuard
        integer, intent(IN)                      :: F_nCcVars
        procedure(milhoja_runtime_taskFunction)  :: F_initBlock
        procedure(milhoja_errorEstimateCallback) :: F_errorEst
        integer, intent(OUT)                     :: F_ierr

        integer(MILHOJA_INT) :: C_globalCommF
        real(MILHOJA_REAL)   :: C_xMin, C_xMax
        real(MILHOJA_REAL)   :: C_yMin, C_yMax
        real(MILHOJA_REAL)   :: C_zMin, C_zMax
        integer(MILHOJA_INT) :: C_nxb, C_nyb, C_nzb
        integer(MILHOJA_INT) :: C_nBlocksX, C_nBlocksY, C_nBlocksZ
        integer(MILHOJA_INT) :: C_lRefineMax
        integer(MILHOJA_INT) :: C_nGuard
        integer(MILHOJA_INT) :: C_nCcVars
        type(C_FUNPTR)       :: C_initBlock
        type(C_FUNPTR)       :: C_errorEst
        integer(MILHOJA_INT) :: C_ierr

        integer :: rank
        integer :: mpierr

        ! If the runtime module is used, then this module must also be used.
        ! However, the Milhoja grid unit could be used
        ! without this runtime.  Therefore both should confirm correct
        ! types to be safe, but we have only the grid module print type info.
        CALL milhoja_types_confirmMatchingTypes(F_ierr)
        if (F_ierr /= MILHOJA_SUCCESS) then
            RETURN
        end if

        CALL MPI_Comm_rank(F_globalCommF, rank, mpierr)
        if (rank == MASTER_PE) then
            CALL milhoja_types_printTypesInformation(STDOUT, F_ierr)
            if (F_ierr /= MILHOJA_SUCCESS) then
                RETURN
            end if
        end if

        C_globalCommF =  INT(F_globalCommF, kind=MILHOJA_INT)
        C_xMin        = REAL(F_xMin,        kind=MILHOJA_REAL)
        C_xMax        = REAL(F_xMax,        kind=MILHOJA_REAL)
        C_yMin        = REAL(F_yMin,        kind=MILHOJA_REAL)
        C_yMax        = REAL(F_yMax,        kind=MILHOJA_REAL)
        C_zMin        = REAL(F_zMin,        kind=MILHOJA_REAL)
        C_zMax        = REAL(F_zMax,        kind=MILHOJA_REAL)
        C_nxb         =  INT(F_nxb,         kind=MILHOJA_INT)
        C_nyb         =  INT(F_nyb,         kind=MILHOJA_INT)
        C_nzb         =  INT(F_nzb,         kind=MILHOJA_INT)
        C_nBlocksX    =  INT(F_nBlocksX,    kind=MILHOJA_INT)
        C_nBlocksY    =  INT(F_nBlocksY,    kind=MILHOJA_INT)
        C_nBlocksZ    =  INT(F_nBlocksZ,    kind=MILHOJA_INT)
        C_lRefineMax  =  INT(F_lRefineMax,  kind=MILHOJA_INT)
        C_nGuard      =  INT(F_nGuard,      kind=MILHOJA_INT)
        C_nCcVars     =  INT(F_nCcVars,     kind=MILHOJA_INT)

        C_initBlock   = C_FUNLOC(F_initBlock)
        C_errorEst    = C_FUNLOC(F_errorEst)

        C_ierr = milhoja_grid_init_C(C_globalCommF,                      &
                                     C_xMin, C_xMax,                     &
                                     C_yMin, C_yMax,                     &
                                     C_zMin, C_zMax,                     &
                                     C_nxb, C_nyb, C_nzb,                &
                                     C_nBlocksX, C_nBlocksY, C_nBlocksZ, &
                                     C_lRefineMax,                       &
                                     C_nGuard, C_nCcVars,                &
                                     C_initBlock, C_errorEst)
        F_ierr = INT(C_ierr)
    end subroutine milhoja_grid_init

    !> Finalize the grid infrastructure.  It is assumed that calling code is
    !! responsible for finalizing MPI and does so *after* calling this routine.
    !!
    !! Calling code should finalize the grid before finalizing the runtime.
    !!
    !! @todo Confirm that grid must be finalized first.
    !!
    !! @param F_ierr    The milhoja error code
    subroutine milhoja_grid_finalize(F_ierr)
        integer, intent(OUT) :: F_ierr

        integer(MILHOJA_INT) :: C_ierr

        C_ierr = milhoja_grid_finalize_C()
        F_ierr = INT(C_ierr)
    end subroutine milhoja_grid_finalize

    !> Initialize the domain and set the initial conditions such that the mesh
    !! refinement across the domain is consistent with the initial conditions.
    !!
    !! The initial conditions are presently computed and stored using the 
    !! routine given during initialization and using the runtime with the
    !! CPU-only thread team configuration.
    !!
    !! @todo Allow for not using the runtime to set ICs or to use a different
    !!       TT config such as the GPU-only or CPU/GPU data parallel configs.
    !!
    !! @param F_nDistributorThreads  The number of distributor threads to 
    !!                               activate in the CPU-only thread team
    !!                               configuration
    !! @param F_nTeamThreads         The number of threads to activate in the
    !!                               single thread team 
    !! @param F_ierr                 The milhoja error code
    subroutine milhoja_grid_initDomain(F_nDistributorThreads, &
                                       F_nTeamThreads,        &
                                       F_ierr)
        integer, intent(IN)  :: F_nDistributorThreads
        integer, intent(IN)  :: F_nTeamThreads
        integer, intent(OUT) :: F_ierr

        integer(MILHOJA_INT) :: C_nDistributorThreads
        integer(MILHOJA_INT) :: C_nTeamThreads
        integer(MILHOJA_INT) :: C_ierr

        C_nDistributorThreads = INT(F_nDistributorThreads, kind=MILHOJA_INT)
        C_nTeamThreads        = INT(F_nTeamThreads,        kind=MILHOJA_INT)

        C_ierr = milhoja_grid_init_domain_C(C_nDistributorThreads, &
                                            C_nTeamThreads)
        F_ierr = INT(C_ierr)
    end subroutine milhoja_grid_initDomain

    !> Obtain the index of the finest mesh refinement level that could be
    !! used at any time during execution.
    !!
    !! @param F_level    The 1-based index of the level where 1 is coarsest
    !! @param F_ierr     The milhoja error code
    subroutine milhoja_grid_getMaxFinestLevel(F_level, F_ierr)
        integer, intent(OUT) :: F_level
        integer, intent(OUT) :: F_ierr

        integer(MILHOJA_INT) :: C_level
        integer(MILHOJA_INT) :: C_ierr

        C_ierr = milhoja_grid_max_finest_level_C(C_level)
        F_ierr = INT(C_ierr)

        ! Assuming C interface has 1-based level index set
        F_level = INT(C_level)
    end subroutine milhoja_grid_getMaxFinestLevel

    !> Obtain the index of the finest mesh refinement level that is currently
    !! in existence and use.
    !!
    !! @param F_level    The 1-based index of the level where 1 is coarsest
    !! @param F_ierr     The milhoja error code
    subroutine milhoja_grid_getCurrentFinestLevel(F_level, F_ierr)
        integer, intent(OUT) :: F_level
        integer, intent(OUT) :: F_ierr

        integer(MILHOJA_INT) :: C_level
        integer(MILHOJA_INT) :: C_ierr

        C_ierr = milhoja_grid_current_finest_level_C(C_level)
        F_ierr = INT(C_ierr)

        ! Assuming C interface has 1-based level index set
        F_level = INT(C_level)
    end subroutine milhoja_grid_getCurrentFinestLevel

    !> Obtain the low and high coordinates in physical space of the rectangular
    !! box that bounds the problem's spatial domain.
    !!
    !! NOTE: This routine does not presume to know what values to set for
    !! coordinate components above NDIM.  Therefore, calling code is responsible
    !! for setting or ignoring such data.  This routine will not alter or
    !! overwrite such values in the given arrays.
    !!
    !! @param F_lo      The coordinates of the low point used to define the box
    !! @param F_hi      The coordinates of the high point used to define the box
    !! @param F_ierr    The milhoja error code
    subroutine milhoja_grid_getDomainBoundBox(F_lo, F_hi, F_ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_LOC

        real,    intent(INOUT), target :: F_lo(1:MDIM)
        real,    intent(INOUT), target :: F_hi(1:MDIM)
        integer, intent(OUT)           :: F_ierr

        type(C_PTR)          :: C_lo
        type(C_PTR)          :: C_hi
        integer(MILHOJA_INT) :: C_ierr

        C_lo = C_LOC(F_lo)
        C_hi = C_LOC(F_hi)
        C_ierr = milhoja_grid_domain_bound_box_C(C_lo, C_hi)
        F_ierr = INT(C_ierr)
    end subroutine milhoja_grid_getDomainBoundBox

    !> Obtain the mesh refinement values for the given refinement level.
    !!
    !! NOTE: This routine does not presume to know what values to set for
    !! resolution values above NDIM.  Therefore, calling code is responsible
    !! for setting or ignoring such data.  This routine will not alter or
    !! overwrite such values in the given array.
    !!
    !! @param F_level   The 1-based index of the level of interest with 1
    !!                  being the coarsest level
    !! @param F_deltas  The mesh resolution values
    !! @param F_ierr    The milhoja error code
    subroutine milhoja_grid_getDeltas(F_level, F_deltas, F_ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_LOC

        integer, intent(IN)            :: F_level
        real,    intent(INOUT), target :: F_deltas(1:MDIM)
        integer, intent(OUT)           :: F_ierr

        integer(MILHOJA_INT) :: C_level
        type(C_PTR)          :: C_deltas
        integer(MILHOJA_INT) :: C_ierr

        ! Assuming C interface has 1-based level index set
        C_level = INT(F_level, kind=MILHOJA_INT)

        C_deltas = C_LOC(F_deltas)
        C_ierr = milhoja_grid_deltas_C(C_level, C_deltas)
        F_ierr = INT(C_ierr)
    end subroutine milhoja_grid_getDeltas

    !> Write the contents of the solution to file.  It is intended that this
    !! routine only be used for development, testing, and debugging.
    !!
    !! Refer to the code to determine the format of the created file.
    !! The data is written to a file name
    !!      milhoja_plt_<F_nstep>
    !!
    !! @todo Calling code should pass in the full filename and not the step
    !! 
    !! @param F_step   The number of the timestep associated with the data
    !! @param F_ierr   The milhoja error code
    subroutine milhoja_grid_writePlotfile(F_step, F_ierr)
        integer, intent(IN)  :: F_step
        integer, intent(OUT) :: F_ierr

        integer(MILHOJA_INT) :: C_step
        integer(MILHOJA_INT) :: C_ierr

        C_step = INT(F_step, kind=MILHOJA_INT)

        C_ierr = milhoja_grid_write_plotfile_C(C_step)
        F_ierr = INT(C_ierr)
    end subroutine milhoja_grid_writePlotfile

end module milhoja_grid_mod

