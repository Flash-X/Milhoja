!> A module in the Milhoja Fortran/C++ interoperability layer that provides
!! calling code with a high-level Fortran interface over tile objects.
!!
!! Generally, tiles are not created in calling code.  Rather, calling code is
!! given access to a tile as an argument to a runtime task function or via
!! grid infrastructure such as a tile iterator.  It is assumed that upon being
!! given a tile, the calling code will store the underlying C dataItem pointer
!! and use this to acquire the tile's data through this interface.  In this
!! sense, the pointer is effectively an index for the tile.
!!
!! @todo Continue building out as actual Flash-X use cases require more
!!       metadata.

#include "milhoja_interface_error_codes.h"

#include "Flash.h"
#include "constants.h"

module milhoja_tile_mod
    use milhoja_types_mod, ONLY : MILHOJA_INT

    implicit none
    private

    !!!!!----- PUBLIC INTERFACE
    public :: milhoja_tile_getMetadata

    !!!!!----- INTERFACES TO C-LINKAGE C++ FUNCTIONS
    ! The C-to-Fortran interoperability layer
    interface
        !> Fortran interface on routine in C interface of same name.
        !! Refer to documentation of C routine for more information.
        function milhoja_tile_get_metadata_c(C_dataItemPtr,              &
                                             C_gId, C_level,             &
                                             C_lo, C_hi, C_loGC, C_hiGC, &
                                             C_nVariables, C_dataPtr)    &
                                             result(C_ierr) bind(c)
            use iso_c_binding,     ONLY : C_PTR
            use milhoja_types_mod, ONLY : MILHOJA_INT
            type(C_PTR),          intent(IN), value :: C_dataItemPtr
            integer(MILHOJA_INT), intent(OUT)       :: C_gId
            integer(MILHOJA_INT), intent(OUT)       :: C_level
            type(C_PTR),          intent(IN), value :: C_lo
            type(C_PTR),          intent(IN), value :: C_hi
            type(C_PTR),          intent(IN), value :: C_loGC
            type(C_PTR),          intent(IN), value :: C_hiGC
            integer(MILHOJA_INT), intent(OUT)       :: C_nVariables
            type(C_PTR),          intent(OUT)       :: C_dataPtr
            integer(MILHOJA_INT)                    :: C_ierr
        end function milhoja_tile_get_metadata_c
    end interface

contains

    !> Obtain basic, generally-useful metadata for the given tile.  This is
    !! eager acquisition of only metadata that we assume will need to be
    !! acquired every time a tile is used.  We avoid eager acquisition of
    !! metadata that might need to be used only seldomly.
    !!
    !! NOTE: This routine does not presume to know what values to set for
    !! coordinate components above NDIM.  Therefore, calling code is responsible
    !! for setting or ignoring such data.  This routine will not alter or
    !! overwrite such values in the given arrays.
    !!
    !! The metadata includes a Fortran pointer mapped onto the tile's data in
    !! host memory and reinterpreted as a 4D data array.  For spatial dimensions
    !! above NDIM, the size is 1.  The bounds of the spatial indices as well as
    !! the variable index all start at 1.  Calling code is reponsible for
    !! shifting the bounds as necessary.
    !!
    !! This routine gives access to the tile's index in the underlying
    !! grid backend's index set.  For Paramesh, this would just be a
    !! blockID.  For AMReX, the index is (grid index, tile index, level).
    !! This means that this routine could be different for each backend,
    !! which is not desired.  For now, we are content with carrying around
    !! all data used to derive tile indices across all backends.  All such
    !! data can be passed back down to the concrete backend implemention
    !! and the backend can pick out the information that it needs to 
    !! uniquely identify the tile.  Can kicked down the road...
    !!
    !! @todo Use Klaus's macro for declaring dummy pointer intent
    !! @todo Are the other indices really necessary or is the C pointer
    !!       sufficient?  Presumably, once the pointer is passed into the C++
    !!       code, the pointer could be used to access the backend-specific
    !!       index.  At the very least having the other indices
    !!       might be useful for developing/debugging.  If not necessary,
    !!       move these out to a debugging routine that can be used to
    !!       acquire them only when necessary.
    !! @todo See if we can immediately
    !!       set the given F pointer to have its bounds in the global index
    !!       space.
    !! @todo The layer should not write to stdout/stderr directly.  Error
    !!       messages should be cached with a Fortran/C++ milhoja error manager
    !!       so that calling code can retrieve the message and use as desired.
    !!
    !! @param C_dataItemPtr   The pointer to the C++ dataItem (i.e. tile) object
    !! @param gId             The grid index associated with the tile, which is
    !!                        presumably meaningful to the grid backend
    !! @param level           The 1-based index of the tile's refinement level
    !! @param lo              The global indices (1-based) of the low point that
    !!                        defines the interior of the tile
    !! @param hi              The global indices (1-based) of the high point that
    !!                        defines the interior of the tile
    !! @param loGC            The global indices (1-based) of the low point that
    !!                        defines the interior+GC of the tile
    !! @param hiGC            The global indices (1-based) of the high point that
    !!                        defines the interior+GC of the tile
    !! @param C_dataPtr       The C data pointer reinterpreted as a C_PTR derived type
    !! @param F_dataPtr       The C data pointer reinterpreted as a Fortran array.
    !! @param ierr            The milhoja error code
    subroutine milhoja_tile_getMetadata(C_dataItemPtr,        &
                                        gId, level,           &
                                        lo, hi, loGC, hiGC,   &
                                        C_dataPtr, F_dataPtr, &
                                        ierr)
        use iso_c_binding,      ONLY : C_PTR, &
                                       C_LOC, &
                                       C_ASSOCIATED, &
                                       C_F_POINTER

        use milhoja_types_mod, ONLY : MILHOJA_REAL

        type(C_PTR),          intent(IN)                    :: C_dataItemPtr
        integer(MILHOJA_INT), intent(OUT)                   :: gId
        integer(MILHOJA_INT), intent(OUT)                   :: level
        integer(MILHOJA_INT), intent(INOUT), target         :: lo(1:MDIM)
        integer(MILHOJA_INT), intent(INOUT), target         :: hi(1:MDIM)
        integer(MILHOJA_INT), intent(INOUT), target         :: loGC(1:MDIM)
        integer(MILHOJA_INT), intent(INOUT), target         :: hiGC(1:MDIM)
        type(C_PTR),          intent(INOUT)                 :: C_dataPtr
        real(MILHOJA_REAL),                         pointer :: F_dataPtr(:, :, :, :) ! intent(INOUT)
        integer(MILHOJA_INT), intent(OUT)                   :: ierr

        type(C_PTR)          :: lo_CPTR
        type(C_PTR)          :: hi_CPTR
        type(C_PTR)          :: loGC_CPTR
        type(C_PTR)          :: hiGC_CPTR
        integer(MILHOJA_INT) :: nVariables

        integer :: dataShape(1:MDIM+1)

        gId = -1
        level = -1

        ! Prevent possible memory leaks
        if      (C_ASSOCIATED(C_dataPtr)) then
            write(*,'(A)') "[milhoja_tile_getMetadata] C_dataPtr not NULL"
            ierr = MILHOJA_ERROR_POINTER_NOT_NULL
            RETURN
        else if (ASSOCIATED(F_dataPtr)) then
            write(*,'(A)') "[milhoja_tile_getMetadata] F_dataPtr not NULL"
            ierr = MILHOJA_ERROR_POINTER_NOT_NULL
            RETURN
        end if

        ! Assuming for C interface that points are defined w.r.t. a 1-based
        ! global index space.
        lo_CPTR   = C_LOC(lo)
        hi_CPTR   = C_LOC(hi)
        loGC_CPTR = C_LOC(loGC)
        hiGC_CPTR = C_LOC(hiGC)

        ! Assuming for C interface that level index set is 1-based
        ierr = milhoja_tile_get_metadata_c(C_dataItemPtr,           &
                                           gId, level,              &
                                           lo_CPTR,   hi_CPTR,      &
                                           loGC_CPTR, hiGC_CPTR,    &
                                           nVariables, C_dataPtr)

        if (ierr == MILHOJA_SUCCESS) then
            if (.NOT. C_ASSOCIATED(C_dataPtr)) then
                write(*,'(A)') "[milhoja_tile_getMetaData] C_dataPtr is NULL"
                ierr = MILHOJA_ERROR_POINTER_IS_NULL
                RETURN
            end if

            dataShape(1:MDIM) = hiGC(1:MDIM) - loGC(1:MDIM) + 1
            dataShape(MDIM+1) = nVariables
            CALL C_F_POINTER(C_dataPtr, F_dataPtr, shape=dataShape)
            if (.NOT. ASSOCIATED(F_dataPtr)) then
                write(*,'(A)') "[milhoja_tile_getMetadata] F_dataPtr is NULL"
                ierr = MILHOJA_ERROR_POINTER_IS_NULL
                RETURN
            end if
        end if
    end subroutine milhoja_tile_getMetadata

end module milhoja_tile_mod

