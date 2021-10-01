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
    !! @todo Does it matter that we convert an MPI Comm from a default integer
    !!       to a potentially different kind?  Does MPI have the flexibility 
    !!       to deal with that?
    !!
    !! @param globalCommF  The Fortran version of the MPI communicator that
    !!                     Milhoja should use
    !! @param xMin         Define the physical domain in X as [F_xMin, F_xMax]
    !! @param xMax         Define the physical domain in X as [F_xMin, F_xMax]
    !! @param yMin         Define the physical domain in Y as [F_yMin, F_yMax]
    !! @param yMax         Define the physical domain in Y as [F_yMin, F_yMax]
    !! @param zMin         Define the physical domain in Z as [F_zMin, F_zMax]
    !! @param zMax         Define the physical domain in Z as [F_zMin, F_zMax]
    !! @param nxb          The number of cells along X in each block in the
    !!                     domain decomposition
    !! @param nyb          The number of cells along Y in each block in the
    !!                     domain decomposition
    !! @param nzb          The number of cells along Z in each block in the
    !!                     domain decomposition
    !! @param nBlocksX     The number of blocks along X in the domain decomposition
    !! @param nBlocksY     The number of blocks along Y in the domain decomposition
    !! @param nBlocksZ     The number of blocks along Z in the domain decomposition
    !! @param lRefineMax   The 1-based index of the finest refinement level
    !!                     permitted at any time during the simulation
    !! @param nGuard       The number of guardcells
    !! @param nCcVars      The number of physical variables in the solution
    !! @param initBlock    Procedure to use to compute and store the initial
    !!                     conditions
    !! @param errorEst     Procedure that is used to assess if a block should
    !!                     be refined, derefined, or stay at the same
    !!                     refinement
    !! @param ierr         The milhoja error code
    subroutine milhoja_grid_init(globalCommF,                  &
                                 xMin, xMax,                   &
                                 yMin, yMax,                   &
                                 zMin, zMax,                   &
                                 nxb, nyb, nzb,                &
                                 nBlocksX, nBlocksY, nBlocksZ, &
                                 lRefineMax,                   &
                                 nGuard, nCcVars,              &
                                 initBlock, errorEst,          &
                                 ierr)
        use iso_c_binding, ONLY : C_FUNPTR, &
                                  C_FUNLOC

        use milhoja_runtime_mod, ONLY : milhoja_runtime_taskFunction

        integer(MILHOJA_INT),                    intent(IN) :: globalCommF
        real(MILHOJA_REAL),                      intent(IN) :: xMin, xMax
        real(MILHOJA_REAL),                      intent(IN) :: yMin, yMax
        real(MILHOJA_REAL),                      intent(IN) :: zMin, zMax
        integer(MILHOJA_INT),                    intent(IN) :: nxb, nyb, nzb
        integer(MILHOJA_INT),                    intent(IN) :: nBlocksX, nBlocksY, nBlocksZ
        integer(MILHOJA_INT),                    intent(IN) :: lRefineMax
        integer(MILHOJA_INT),                    intent(IN) :: nGuard
        integer(MILHOJA_INT),                    intent(IN) :: nCcVars
        procedure(milhoja_runtime_taskFunction)             :: initBlock
        procedure(milhoja_errorEstimateCallback)            :: errorEst
        integer(MILHOJA_INT), intent(OUT)                   :: ierr

        type(C_FUNPTR) :: initBlock_CPTR
        type(C_FUNPTR) :: errorEst_CPTR

        initBlock_CPTR = C_FUNLOC(initBlock)
        errorEst_CPTR  = C_FUNLOC(errorEst)

        ierr = milhoja_grid_init_C(globalCommF,                  &
                                   xMin, xMax,                   &
                                   yMin, yMax,                   &
                                   zMin, zMax,                   &
                                   nxb, nyb, nzb,                &
                                   nBlocksX, nBlocksY, nBlocksZ, &
                                   lRefineMax,                   &
                                   nGuard, nCcVars,              &
                                   initBlock_CPTR, errorEst_CPTR)
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
    !! @param nDistributorThreads  The number of distributor threads to 
    !!                             activate in the CPU-only thread team
    !!                             configuration
    !! @param nTeamThreads         The number of threads to activate in the
    !!                             single thread team 
    !! @param ierr                 The milhoja error code
    subroutine milhoja_grid_initDomain(nDistributorThreads, &
                                       nTeamThreads,        &
                                       ierr)
        integer(MILHOJA_INT), intent(IN)  :: nDistributorThreads
        integer(MILHOJA_INT), intent(IN)  :: nTeamThreads
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ierr = milhoja_grid_init_domain_C(nDistributorThreads, nTeamThreads)
    end subroutine milhoja_grid_initDomain

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

    !> Obtain the low and high coordinates in physical space of the rectangular
    !! box that bounds the problem's spatial domain.
    !!
    !! NOTE: This routine does not presume to know what values to set for
    !! coordinate components above NDIM.  Therefore, calling code is responsible
    !! for setting or ignoring such data.  This routine will not alter or
    !! overwrite such values in the given arrays.
    !!
    !! @param lo      The coordinates of the low point used to define the box
    !! @param hi      The coordinates of the high point used to define the box
    !! @param ierr    The milhoja error code
    subroutine milhoja_grid_getDomainBoundBox(lo, hi, ierr)
        use iso_c_binding, ONLY : C_PTR, &
                                  C_LOC

        real(MILHOJA_REAL),   intent(INOUT), target :: lo(1:MDIM)
        real(MILHOJA_REAL),   intent(INOUT), target :: hi(1:MDIM)
        integer(MILHOJA_INT), intent(OUT)           :: ierr

        type(C_PTR) :: lo_CPTR
        type(C_PTR) :: hi_CPTR

        lo_CPTR = C_LOC(lo)
        hi_CPTR = C_LOC(hi)
        ierr = milhoja_grid_domain_bound_box_C(lo_CPTR, hi_CPTR)
    end subroutine milhoja_grid_getDomainBoundBox

    !> Obtain the mesh refinement values for the given refinement level.
    !!
    !! NOTE: This routine does not presume to know what values to set for
    !! resolution values above NDIM.  Therefore, calling code is responsible
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
        real(MILHOJA_REAL),   intent(INOUT), target :: deltas(1:MDIM)
        integer(MILHOJA_INT), intent(OUT)           :: ierr

        type(C_PTR) :: deltas_CPTR

        deltas_CPTR = C_LOC(deltas)

        ! Assuming C interface has 1-based level index set
        ierr = milhoja_grid_deltas_C(level, deltas_CPTR)
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
    !! @param step   The number of the timestep associated with the data
    !! @param ierr   The milhoja error code
    subroutine milhoja_grid_writePlotfile(step, ierr)
        integer(MILHOJA_INT), intent(IN)  :: step
        integer(MILHOJA_INT), intent(OUT) :: ierr

        ierr = milhoja_grid_write_plotfile_C(step)
    end subroutine milhoja_grid_writePlotfile

end module milhoja_grid_mod

