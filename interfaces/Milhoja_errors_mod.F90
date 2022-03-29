#include "Milhoja_interface_error_codes.h"

!> A module that Fortran code can use to work with Milhoja errors.
module milhoja_errors_mod

    implicit none
    private

    !> All Milhoja error messages will be no longer than this value.
    integer, parameter, public :: MAX_ERROR_LENGTH = 128

    public :: milhoja_errorMessage

contains

    !> Obtain the error message associated with the given error code.
    !!
    !! \param errorCode   The Milhoja error code of interest
    !! \param errorMsg    The associated error message
    subroutine milhoja_errorMessage(errorCode, errorMsg)
        use milhoja_types_mod, ONLY : MILHOJA_INT

        integer(MILHOJA_INT),            intent(IN)  :: errorCode
        character(LEN=MAX_ERROR_LENGTH), intent(OUT) :: errorMsg

        select case (errorCode)
        case (MILHOJA_SUCCESS)
            errorMsg = "No Milhoja error occurred"
        case (MILHOJA_ERROR_INVALID_LEVEL)
            errorMsg = "Invalid level value"
        case (MILHOJA_ERROR_NEGATIVE_VALUE_FOR_UINT)
            errorMsg = "Negative value given for cast to unsigned int"
        case (MILHOJA_ERROR_POINTER_IS_NULL)
            errorMsg = "Pointer is null"
        case (MILHOJA_ERROR_POINTER_NOT_NULL)
            errorMsg = "Pointer is *not* null"
        case (MILHOJA_ERROR_UNABLE_TO_INIT_GRID)
            errorMsg = "Unable to initialize grid backend"
        case (MILHOJA_ERROR_UNABLE_TO_FINALIZE_GRID)
            errorMsg = "Unable to finalize grid backend"
        case (MILHOJA_ERROR_UNABLE_TO_GET_BOUNDS)
            errorMsg = "Unable to get bounds"
        case (MILHOJA_ERROR_UNABLE_TO_GET_LEVEL)
            errorMsg = "Unable to get level"
        case (MILHOJA_ERROR_UNABLE_TO_GET_DELTAS)
            errorMsg = "Unable to get mesh deltas"
        case (MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN)
            errorMsg = "Unable to initialize the domain"
        case (MILHOJA_ERROR_UNABLE_TO_GET_BLOCK_SIZE)
            errorMsg = "Unable to get the block size in cells/edge"
        case (MILHOJA_ERROR_UNABLE_TO_GET_DOMAIN_DECOMPOSITION)
            errorMsg = "Unable to get the block domain decomposition"
        case (MILHOJA_ERROR_UNABLE_TO_GET_N_GUARDCELLS)
            errorMsg = "Unable to get the number of guardcells"
        case (MILHOJA_ERROR_UNABLE_TO_GET_N_CC_VARS)
            errorMsg = "Unable to get the number of cell-centered variables"
        case (MILHOJA_ERROR_UNABLE_TO_BUILD_ITERATOR)
            errorMsg = "Unable to build an iterator"
        case (MILHOJA_ERROR_UNABLE_TO_DESTROY_ITERATOR)
            errorMsg = "Unable to destroy given iterator"
        case (MILHOJA_ERROR_UNABLE_TO_VALIDATE_ITERATOR)
            errorMsg = "Unable to validate the state of given iterator"
        case (MILHOJA_ERROR_UNABLE_TO_ADVANCE_ITERATOR)
            errorMsg = "Unable to advance given iterator to next tile"
        case (MILHOJA_ERROR_UNABLE_TO_GET_TILE_METADATA)
            errorMsg = "Unable to retrieve metadata of iterators current tile"
        case DEFAULT
            errorMsg = "Unknown Milhoja error code"
        end select
    end subroutine milhoja_errorMessage

end module milhoja_errors_mod

