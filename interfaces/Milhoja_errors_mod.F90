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
    !! \param error_code   The Milhoja error code of interest
    !! \param error_msg    The associated error message
    subroutine milhoja_errorMessage(error_code, error_msg)
        use milhoja_types_mod, ONLY : MILHOJA_INT

        integer(MILHOJA_INT),            intent(IN)  :: error_code
        character(LEN=MAX_ERROR_LENGTH), intent(OUT) :: error_msg

        select case (error_code)
        case (MILHOJA_SUCCESS)
            error_msg = "No Milhoja error occurred"
        case (MILHOJA_ERROR_INVALID_LEVEL)
            error_msg = "Invalid level value"
        case (MILHOJA_ERROR_NEGATIVE_VALUE_FOR_UINT)
            error_msg = "Negative value given for cast to unsigned int"
        case (MILHOJA_ERROR_POINTER_IS_NULL)
            error_msg = "Pointer is null"
        case (MILHOJA_ERROR_UNABLE_TO_INIT_GRID)
            error_msg = "Unable to initialize grid backend"
        case (MILHOJA_ERROR_UNABLE_TO_FINALIZE_GRID)
            error_msg = "Unable to finalize grid backend"
        case (MILHOJA_ERROR_UNABLE_TO_GET_BOUNDS)
            error_msg = "Unable to get bounds"
        case (MILHOJA_ERROR_UNABLE_TO_GET_LEVEL)
            error_msg = "Unable to get level"
        case (MILHOJA_ERROR_UNABLE_TO_GET_DELTAS)
            error_msg = "Unable to get mesh deltas"
        case (MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN)
            error_msg = "Unable to initialize the domain"
        case DEFAULT
            error_msg = "Unknown Milhoja error code"
        end select
    end subroutine milhoja_errorMessage

end module milhoja_errors_mod

