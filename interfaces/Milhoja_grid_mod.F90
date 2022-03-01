#include "Milhoja.h"

!> A module in the Milhoja Fortran/C++ interoperability layer that provides
!! calling code with a high-level Fortran interface for interacting with
!! the the Milhoja grid infrastructure.
!!
!! @todo Build out full interface including iterator, AMR functionality,
!!       and flux correction.
module milhoja_grid_mod
    use milhoja_types_mod, ONLY : MILHOJA_INT, &
                                  MILHOJA_REAL

    implicit none
    private

    !!!!!----- PUBLIC INTERFACE
    public :: milhoja_grid_init
    public :: milhoja_grid_finalize
    public :: milhoja_grid_getDomainBoundBox
    public :: milhoja_grid_getMaxFinestLevel
    public :: milhoja_grid_getCurrentFinestLevel
    public :: milhoja_grid_getDeltas
    public :: milhoja_grid_initDomain

    !!!!!----- FORTRAN INTERFACES TO MILHOJA FUNCTION POINTERS
    abstract interface
        !> Fortran interface of the routine calling code gives the grid
        !! infrastructure so that Milhoja can set the initial conditions in the
        !! given data item.
        subroutine milhoja_initBlock(C_threadID, C_dataItemPtr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_threadID
            type(C_PTR),          intent(IN), value :: C_dataItemPtr
        end subroutine milhoja_initBlock

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
        function milhoja_grid_init_C(F_globalComm,                       &
                                     C_logRank,                          &
                                     C_xMin, C_xMax,                     &
                                     C_yMin, C_yMax,                     &
                                     C_zMin, C_zMax,                     &
                                     C_nxb, C_nyb, C_nzb,                &
                                     C_nBlocksX, C_nBlocksY, C_nBlocksZ, &
                                     C_maxRefinementLevel,               &
                                     C_nGuard, C_nCcVars,                &
                                     C_errorEst) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_FUNPTR
            use mpi,               ONLY : MPI_INTEGER_KIND
            use milhoja_types_mod, ONLY : MILHOJA_INT, MILHOJA_REAL
            implicit none
            integer(MPI_INTEGER_KIND), intent(IN), value :: F_globalComm
            integer(MILHOJA_INT),      intent(IN), value :: C_logRank
            real(MILHOJA_REAL),        intent(IN), value :: C_xMin, C_xMax
            real(MILHOJA_REAL),        intent(IN), value :: C_yMin, C_yMax
            real(MILHOJA_REAL),        intent(IN), value :: C_zMin, C_zMax
            integer(MILHOJA_INT),      intent(IN), value :: C_nxb, C_nyb, C_nzb
            integer(MILHOJA_INT),      intent(IN), value :: C_nBlocksX, C_nBlocksY, C_nBlocksZ
            integer(MILHOJA_INT),      intent(IN), value :: C_maxRefinementLevel
            integer(MILHOJA_INT),      intent(IN), value :: C_nGuard
            integer(MILHOJA_INT),      intent(IN), value :: C_nCcVars
            type(C_FUNPTR),            intent(IN), value :: C_errorEst
            integer(MILHOJA_INT)                         :: C_ierr
        end function milhoja_grid_init_C

        !> Fortran interface on routine in C interface of same name.
        function milhoja_grid_finalize_C() result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT) :: C_ierr
        end function milhoja_grid_finalize_C

        !> Fortran interface on routine in C interface of same name.
        function milhoja_grid_domain_bound_box_C(C_lo, C_hi) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_PTR),         intent(IN), value :: C_lo
            type(C_PTR),         intent(IN), value :: C_hi
            integer(MILHOJA_INT)                   :: C_ierr
        end function milhoja_grid_domain_bound_box_C

        !> Fortran interface on routine in C interface of same name.
        function milhoja_grid_max_finest_level_C(C_level) result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(OUT) :: C_level
            integer(MILHOJA_INT)              :: C_ierr
        end function milhoja_grid_max_finest_level_C

        !> Fortran interface on routine in C interface of same name.
        function milhoja_grid_current_finest_level_C(C_level) result(C_ierr) bind(c)
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(OUT) :: C_level
            integer(MILHOJA_INT)              :: C_ierr
        end function milhoja_grid_current_finest_level_C

        !> Fortran interface on routine in C interface of same name.
        function milhoja_grid_deltas_C(C_level, C_deltas) result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            integer(MILHOJA_INT), intent(IN), value :: C_level
            type(C_PTR),          intent(IN), value :: C_deltas
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_grid_deltas_C

        !> Fortran interface on routine in C interface of same name.
        function milhoja_grid_init_domain_C(C_initBlockPtr) result(C_ierr) &
                                            bind(c)
            use iso_c_binding,     ONLY : C_FUNPTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            implicit none
            type(C_FUNPTR),      intent(IN), value :: C_initBlockPtr
            integer(MILHOJA_INT)                   :: C_ierr
        end function milhoja_grid_init_domain_C
    end interface

contains

    !> Perform all Milhoja initializations needed for calling code to begin
    !! using its grid infrastructure.  Note that calling code must subsequently
    !! call Grid_initDomain when deemed appropriate to setup the problem domain
    !! in the grid infrastructure.  Calling code must initialize MPI before
    !! calling this routine.
    !!
    !! @todo Does this unit or the runtime need to be initialized
    !!       first?  If so, document here and in runtime.
    !!
    !! @param globalCommF          The Fortran version of the MPI communicator that
    !!                             Milhoja should use
    !! @param logRank              The rank in the given communicator of the MPI process
    !!                             that should perform logging duties.
    !! @param xMin                 Define the physical domain in X as [xMin, xMax]
    !! @param xMax                 See xMin
    !! @param yMin                 Define the physical domain in Y as [yMin, yMax]
    !! @param yMax                 See yMin
    !! @param zMin                 Define the physical domain in Z as [zMin, zMax]
    !! @param zMax                 See zMin
    !! @param nxb                  The number of cells along X in each block in the
    !!                             domain decomposition
    !! @param nyb                  The number of cells along Y in each block in the
    !!                             domain decomposition
    !! @param nzb                  The number of cells along Z in each block in the
    !!                             domain decomposition
    !! @param nBlocksX             The number of blocks along X in the domain decomposition
    !! @param nBlocksY             The number of blocks along Y in the domain decomposition
    !! @param nBlocksZ             The number of blocks along Z in the domain decomposition
    !! @param maxRefinementLevel   The 1-based index of the finest refinement level
    !!                             permitted at any time during the simulation
    !! @param nGuard               The number of guardcells
    !! @param nCcVars              The number of physical variables in the solution
    !! @param errorEst             Procedure that is used to assess if a block should
    !!                             be refined, derefined, or stay at the same
    !!                             refinement
    !! @param ierr                 The milhoja error code
    subroutine milhoja_grid_init(globalCommF, logRank,         &
                                 xMin, xMax,                   &
                                 yMin, yMax,                   &
                                 zMin, zMax,                   &
                                 nxb, nyb, nzb,                &
                                 nBlocksX, nBlocksY, nBlocksZ, &
                                 maxRefinementLevel,           &
                                 nGuard, nCcVars,              &
                                 errorEst,                     &
                                 ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_FUNLOC

        integer,                                 intent(IN)  :: globalCommF
        integer(MILHOJA_INT),                    intent(IN)  :: logRank
        real(MILHOJA_REAL),                      intent(IN)  :: xMin, xMax
        real(MILHOJA_REAL),                      intent(IN)  :: yMin, yMax
        real(MILHOJA_REAL),                      intent(IN)  :: zMin, zMax
        integer(MILHOJA_INT),                    intent(IN)  :: nxb, nyb, nzb
        integer(MILHOJA_INT),                    intent(IN)  :: nBlocksX, nBlocksY, nBlocksZ
        integer(MILHOJA_INT),                    intent(IN)  :: maxRefinementLevel
        integer(MILHOJA_INT),                    intent(IN)  :: nGuard
        integer(MILHOJA_INT),                    intent(IN)  :: nCcVars
        procedure(milhoja_errorEstimateCallback)             :: errorEst
        integer(MILHOJA_INT),                    intent(OUT) :: ierr

        type(C_FUNPTR) :: errorEst_CPTR

        errorEst_CPTR  = C_FUNLOC(errorEst)

        ierr = milhoja_grid_init_C(globalCommF, logRank,         &
                                   xMin, xMax,                   &
                                   yMin, yMax,                   &
                                   zMin, zMax,                   &
                                   nxb, nyb, nzb,                &
                                   nBlocksX, nBlocksY, nBlocksZ, &
                                   maxRefinementLevel,           &
                                   nGuard, nCcVars,              &
                                   errorEst_CPTR)
    end subroutine milhoja_grid_init

    !> Finalize the grid infrastructure.  It is assumed that calling code is
    !! responsible for finalizing MPI and does so *after* calling this routine.
    !!
    !! Calling code should finalize the grid before finalizing the runtime.
    !!
    !! @todo Confirm that grid must be finalized first.
    !!
    !! @param ierr    The milhoja error code
    subroutine milhoja_grid_finalize(ierr)
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ierr = milhoja_grid_finalize_C()
    end subroutine milhoja_grid_finalize

    !> Obtain the low and high coordinates in physical space of the rectangular
    !! box that bounds the problem's spatial domain.
    !!
    !! NOTE: This routine does not presume to know what values to set for
    !! coordinate components above MILHOJA_NDIM.  Therefore, calling code is responsible
    !! for setting or ignoring such data.  This routine will not alter or
    !! overwrite such values in the given arrays.
    !!
    !! @param lo      The coordinates of the low point used to define the box
    !! @param hi      The coordinates of the high point used to define the box
    !! @param ierr    The milhoja error code
    subroutine milhoja_grid_getDomainBoundBox(lo, hi, ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_LOC

        real(MILHOJA_REAL),   intent(INOUT), target :: lo(1:MILHOJA_MDIM)
        real(MILHOJA_REAL),   intent(INOUT), target :: hi(1:MILHOJA_MDIM)
        integer(MILHOJA_INT), intent(OUT)           :: ierr

        type(C_PTR) :: lo_CPTR
        type(C_PTR) :: hi_CPTR

        lo_CPTR = C_LOC(lo)
        hi_CPTR = C_LOC(hi)
        ierr = milhoja_grid_domain_bound_box_C(lo_CPTR, hi_CPTR)
    end subroutine milhoja_grid_getDomainBoundBox

    !> Obtain the index of the finest mesh refinement level that could be
    !! used at any time during execution.
    !!
    !! @param level    The 1-based index of the level where 1 is coarsest
    !! @param ierr     The milhoja error code
    subroutine milhoja_grid_getMaxFinestLevel(level, ierr)
        integer(MILHOJA_INT), intent(OUT) :: level
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ! Assuming C interface has 1-based level index set
        ierr = milhoja_grid_max_finest_level_C(level)
    end subroutine milhoja_grid_getMaxFinestLevel

    !> Obtain the index of the finest mesh refinement level that is currently
    !! in existence and use.
    !!
    !! @param level    The 1-based index of the level where 1 is coarsest
    !! @param ierr     The milhoja error code
    subroutine milhoja_grid_getCurrentFinestLevel(level, ierr)
        integer(MILHOJA_INT), intent(OUT) :: level
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ! Assuming C interface has 1-based level index set
        ierr = milhoja_grid_current_finest_level_C(level)
    end subroutine milhoja_grid_getCurrentFinestLevel

    !> Obtain the mesh refinement values for the given refinement level.
    !!
    !! NOTE: This routine does not presume to know what values to set for
    !! resolution values above MILHOJA_NDIM.  Therefore, calling code is responsible
    !! for setting or ignoring such data.  This routine will not alter or
    !! overwrite such values in the given array.
    !!
    !! @param level   The 1-based index of the level of interest with 1
    !!                being the coarsest level
    !! @param deltas  The mesh resolution values
    !! @param ierr    The milhoja error code
    subroutine milhoja_grid_getDeltas(level, deltas, ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_LOC

        integer(MILHOJA_INT), intent(IN)            :: level
        real(MILHOJA_REAL),   intent(INOUT), target :: deltas(1:MILHOJA_MDIM)
        integer(MILHOJA_INT), intent(OUT)           :: ierr

        type(C_PTR) :: deltas_CPTR

        deltas_CPTR = C_LOC(deltas)

        ! Assuming C interface has 1-based level index set
        ierr = milhoja_grid_deltas_C(level, deltas_CPTR)
    end subroutine milhoja_grid_getDeltas

    !> Initialize the domain and set the initial conditions such that the mesh
    !! refinement across the domain is consistent with the initial conditions.
    !!
    !! This routine applies the initial conditions within each MPI process on a
    !! per-tile basis *without* using the runtime.
    !!
    !! @param initBlock    Procedure to use to compute and store the initial
    !!                     conditions on a single tile
    !! @param ierr         The milhoja error code
    subroutine milhoja_grid_initDomain(initBlock, ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_FUNLOC

        procedure(milhoja_initBlock)             :: initBlock
        integer(MILHOJA_INT),        intent(OUT) :: ierr

        type(C_FUNPTR) :: initBlock_CPTR

        initBlock_CPTR = C_FUNLOC(initBlock)
        ierr = milhoja_grid_init_domain_C(initBlock_CPTR)
    end subroutine milhoja_grid_initDomain

end module milhoja_grid_mod

