!> A module in the Milhoja Fortran/C++ interoperability layer that declares the
!! Fortran types used in the layer and exposed to calling code.  The module
!! includes routines to help calling code work with and understand these these
!! types correctly.
!!
!! All Fortran routines in the Milhoja high-level Fortran interface shall use
!! the integer and real kinds defined in this module as opposed to the
!! C_* kinds in iso_c_binding.
!!
!! The Milhoja Fortran/C++ interoperability layer assumes that calling code will
!! use default Fortran integer and real types and (most) all arguments in the
!! interfaces of routines in the layer are of these types.  This layer is, 
!! therefore, responsible for casting between these default types and those
!! types exposed through the C interface.  This casting is programmed
!! explicitly throughout this layer.  For the sake of simplicity, however,
!! this module can be used to confirm that the default Fortran types as chosen by
!! the calling code and C types are in fact identical so that no error checking
!! of these casts need be coded.
!!
!! NOTE: Several routines in the module use SIZEOF, which is not part of the
!!       Fortran standard.
!! NOTE: The code will likely fail if MILHOJA_INT is *not* set to C_INT.  This
!!       is due to the fact that the C++ code is written with int/unsigned int
!!       rather than a dedicated type that can be easily changed.
!!
!! @todo Get type sizes in standard-conforming way?
!! @todo Once this Fortran interface is moved to the runtime repo and built into
!!       the runtime library, it seems that this assumption should no longer 
!!       be made.  Rather the Fortran interface should expose only the runtime's
!!       types.  This means that this layer becomes simpler as casting is no
!!       longer required.  Calling code will then be responsible for ensuring
!!       that they use this interface correctly, which seems to be more flexible
!!       are reasonable.
!! @todo Should we allow for the int and doubles to be changed at compile time?
!!       Should the type parameter lines be inserted into a template of this
!!       module by the setup tool?
!! @todo Perform type checks against type of library implementation and print
!!       those types as well?

#include "milhoja_interface_error_codes.h"

module milhoja_types_mod
    use IEEE_ARITHMETIC, ONLY : IEEE_SELECTED_REAL_KIND
    use iso_c_binding,   ONLY : C_INT, &
                                C_SIZE_T, &
                                C_DOUBLE

    implicit none
    private

    integer, parameter :: i32 = selected_int_kind(9)
    integer, parameter :: i64 = selected_int_kind(18)
    integer, parameter :: sp = IEEE_SELECTED_REAL_KIND(p=6,  r=37)
    integer, parameter :: dp = IEEE_SELECTED_REAL_KIND(p=15, r=307)

    !!!!!----- PUBLIC INTERFACE
    integer, parameter, public :: MILHOJA_INT    = C_INT
    integer, parameter, public :: MILHOJA_SIZE_T = C_SIZE_T
    integer, parameter, public :: MILHOJA_REAL   = C_DOUBLE

    public :: milhoja_types_confirmMatchingTypes
    public :: milhoja_types_printTypesInformation

contains

    !> Confirm that the Milhoja integer and real types match the default
    !! Fortran integer and real types.
    !!
    !! @todo What about the size_t?
    !!
    !! @param F_ierr  The milhoja error code
    subroutine milhoja_types_confirmMatchingTypes(F_ierr)
        integer, intent(OUT) :: F_ierr

        CALL check_integer_types(F_ierr)
        if (F_ierr /= MILHOJA_SUCCESS) then
            RETURN
        end if

        CALL check_real_types(F_ierr)
    end subroutine milhoja_types_confirmMatchingTypes

    !> Print integer, size_t, and real type information to the given
    !! unit.
    !!
    !! Apparently, the Fortran standard implicitly defines the integer model as
    !! [-HUGE, HUGE].  In particular, it does not include -HUGE-1 in the model
    !! despite the fact that it is built into two's complement and is
    !! implemented as expected on some systems.  This is due to the fact that
    !! -HUGE-1 is a "special" number whose use should likely be avoided.  For
    !! example,
    !!                     -(-HUGE-1) == (-HUGE-1)
    !!      ABS((-HUGE-1) = (-HUGE-1) < 0
    !! Therefore, this routine defines MIN for integer model as -HUGE.
    !!
    !! @param unitId  The unit to which information should be written
    !! @param F_ierr  The milhoja error code
    subroutine milhoja_types_printTypesInformation(unitId, F_ierr)
        integer, intent(IN)  :: unitId
        integer, intent(OUT) :: F_ierr

        integer(i32)            :: int_32
        integer(i64)            :: int_64
        integer                 :: int_default
        integer(MILHOJA_INT)    :: int_milhoja
        integer(MILHOJA_SIZE_T) :: int_size_t

        real(sp)                :: real_sp
        real(dp)                :: real_dp
        real                    :: real_default
        real(MILHOJA_REAL)      :: real_milhoja

        ! GNU v7.5.0 complained when I tried to implement the min value as
        ! -HUGE(*) - 1. 
        ! Error: Integer outside symmetric range implied by Standard Fortran
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A)')          '[Milhoja] Integer Type Characteristics'
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja] SIZEOF(i32)               ', SIZEOF(int_32)
        write(unitId,'(A,I24)')      '[Milhoja]    Max(i32)               ',   HUGE(int_32)
        write(unitId,'(A,I24)')      '[Milhoja]    Min(i32)               ',  -HUGE(int_32)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja] SIZEOF(i64)               ', SIZEOF(int_64)
        write(unitId,'(A,I24)')      '[Milhoja]    Max(i64)               ',   HUGE(int_64)
        write(unitId,'(A,I24)')      '[Milhoja]    Min(i64)               ',  -HUGE(int_64)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja] SIZEOF(default)           ', SIZEOF(int_default)
        write(unitId,'(A,I24)')      '[Milhoja]    Max(default)           ',   HUGE(int_default)
        write(unitId,'(A,I24)')      '[Milhoja]    Min(default)           ',  -HUGE(int_default)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja] SIZEOF(MILHOJA_INT)       ', SIZEOF(int_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja]    Max(MILHOJA_INT)       ',   HUGE(int_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja]    Min(MILHOJA_INT)       ',  -HUGE(int_milhoja)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja] SIZEOF(MILHOJA_SIZE_T)    ', SIZEOF(int_size_t)
        write(unitId,'(A,I24)')      '[Milhoja]    Max(MILHOJA_SIZE_T)    ',   HUGE(int_size_t)
        write(unitId,'(A,I24)')      '[Milhoja]    Min(MILHOJA_SIZE_T)    ',  -HUGE(int_size_t)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A)')          '[Milhoja] Real Type Characteristics'
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja]        KIND(ieee_sp)      ',        KIND(real_sp)
        write(unitId,'(A,I24)')      '[Milhoja]      SIZEOF(ieee_sp)      ',      SIZEOF(real_sp)
        write(unitId,'(A,I24)')      '[Milhoja]   PRECISION(ieee_sp)      ',   PRECISION(real_sp)
        write(unitId,'(A,I24)')      '[Milhoja]       RANGE(ieee_sp)      ',       RANGE(real_sp)
        write(unitId,'(A,I24)')      '[Milhoja]       RADIX(ieee_sp)      ',       RADIX(real_sp)
        write(unitId,'(A,I24)')      '[Milhoja] MINEXPONENT(ieee_sp)      ', MINEXPONENT(real_sp)
        write(unitId,'(A,I24)')      '[Milhoja] MAXEXPONENT(ieee_sp)      ', MAXEXPONENT(real_sp)
        write(unitId,'(A,I24)')      '[Milhoja]      DIGITS(ieee_sp)      ',      DIGITS(real_sp)
        write(unitId,'(A,ES24.15)')  '[Milhoja]     EPSILON(ieee_sp)      ',     EPSILON(real_sp)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        HUGE(ieee_sp)      ',        HUGE(real_sp)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        TINY(ieee_sp)      ',        TINY(real_sp)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja]        KIND(ieee_dp)      ',        KIND(real_dp)
        write(unitId,'(A,I24)')      '[Milhoja]      SIZEOF(ieee_dp)      ',      SIZEOF(real_dp)
        write(unitId,'(A,I24)')      '[Milhoja]   PRECISION(ieee_dp)      ',   PRECISION(real_dp)
        write(unitId,'(A,I24)')      '[Milhoja]       RANGE(ieee_dp)      ',       RANGE(real_dp)
        write(unitId,'(A,I24)')      '[Milhoja]       RADIX(ieee_dp)      ',       RADIX(real_dp)
        write(unitId,'(A,I24)')      '[Milhoja] MINEXPONENT(ieee_dp)      ', MINEXPONENT(real_dp)
        write(unitId,'(A,I24)')      '[Milhoja] MAXEXPONENT(ieee_dp)      ', MAXEXPONENT(real_dp)
        write(unitId,'(A,I24)')      '[Milhoja]      DIGITS(ieee_dp)      ',      DIGITS(real_dp)
        write(unitId,'(A,ES24.15)')  '[Milhoja]     EPSILON(ieee_dp)      ',     EPSILON(real_dp)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        HUGE(ieee_dp)      ',        HUGE(real_dp)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        TINY(ieee_dp)      ',        TINY(real_dp)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja]        KIND(default)      ',        KIND(real_default)
        write(unitId,'(A,I24)')      '[Milhoja]      SIZEOF(default)      ',      SIZEOF(real_default)
        write(unitId,'(A,I24)')      '[Milhoja]   PRECISION(default)      ',   PRECISION(real_default)
        write(unitId,'(A,I24)')      '[Milhoja]       RANGE(default)      ',       RANGE(real_default)
        write(unitId,'(A,I24)')      '[Milhoja]       RADIX(default)      ',       RADIX(real_default)
        write(unitId,'(A,I24)')      '[Milhoja] MINEXPONENT(default)      ', MINEXPONENT(real_default)
        write(unitId,'(A,I24)')      '[Milhoja] MAXEXPONENT(default)      ', MAXEXPONENT(real_default)
        write(unitId,'(A,I24)')      '[Milhoja]      DIGITS(default)      ',      DIGITS(real_default)
        write(unitId,'(A,ES24.15)')  '[Milhoja]     EPSILON(default)      ',     EPSILON(real_default)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        HUGE(default)      ',        HUGE(real_default)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        TINY(default)      ',        TINY(real_default)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'
        write(unitId,'(A,I24)')      '[Milhoja]        KIND(MILHOJA_REAL) ',        KIND(real_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja]      SIZEOF(MILHOJA_REAL) ',      SIZEOF(real_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja]   PRECISION(MILHOJA_REAL) ',   PRECISION(real_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja]       RANGE(MILHOJA_REAL) ',       RANGE(real_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja]       RADIX(MILHOJA_REAL) ',       RADIX(real_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja] MINEXPONENT(MILHOJA_REAL) ', MINEXPONENT(real_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja] MAXEXPONENT(MILHOJA_REAL) ', MAXEXPONENT(real_milhoja)
        write(unitId,'(A,I24)')      '[Milhoja]      DIGITS(MILHOJA_REAL) ',      DIGITS(real_milhoja)
        write(unitId,'(A,ES24.15)')  '[Milhoja]     EPSILON(MILHOJA_REAL) ',     EPSILON(real_milhoja)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        HUGE(MILHOJA_REAL) ',        HUGE(real_milhoja)
        write(unitId,'(A,ES24.15)')  '[Milhoja]        TINY(MILHOJA_REAL) ',        TINY(real_milhoja)
        write(unitId,'(A)')          '[Milhoja] --------------------------------------------------'

        F_ierr = MILHOJA_SUCCESS
    end subroutine milhoja_types_printTypesInformation

    !> Confirm that the Milhoja integer type matches the default Fortran integer
    !! type.
    !!
    !! @param ierr  The milhoja error code
    subroutine check_integer_types(ierr)
        integer, intent(OUT) :: ierr

        integer(i32)         :: int_32
        integer(i64)         :: int_64
        integer              :: int_default
        integer(MILHOJA_INT) :: int_milhoja

        !!!!!----- 32-bit & 64-bit integer
        ! Sanity check since we print info for these types
        if      (SIZEOF(int_32) /= 4) then
            ierr = MILHOJA_ERROR_BAD_I32_SIZE
            RETURN
        else if (SIZEOF(int_64) /= 8) then
            ierr = MILHOJA_ERROR_BAD_I64_SIZE
            RETURN
        end if

        !!!!!----- Confirm that mapped types are same
        if      (SIZEOF(int_default) /= SIZEOF(int_milhoja)) then
            ierr = MILHOJA_ERROR_INT_SIZE_MISMATCH
            RETURN
        else if (HUGE(int_default)   /= HUGE(int_milhoja)) then
            ierr = MILHOJA_ERROR_INT_MAX_MISMATCH
            RETURN
        end if

        ierr = MILHOJA_SUCCESS
    end subroutine check_integer_types

    !> Confirm that the Milhoja real type matches the default Fortran real type.
    !!
    !! @param ierr  The milhoja error code
    subroutine check_real_types(ierr)
        integer, intent(OUT) :: ierr

        real(sp)           :: real_sp
        real(dp)           :: real_dp
        real               :: real_default
        real(MILHOJA_REAL) :: real_milhoja

        !!!!!----- IEEE single precision
        ! Sanity check since we print info for this type
        if      (SIZEOF(real_sp) /= 4) then
            ierr = MILHOJA_ERROR_BAD_FP32_SIZE
            RETURN
        else if (EPSILON(real_sp) /= 1.1920929E-07_sp) then
            ierr = MILHOJA_ERROR_BAD_FP32_EPSILON
            RETURN
        else if (DIGITS(real_sp) /= 24) then
            ierr = MILHOJA_ERROR_BAD_FP32_DIGITS
            RETURN
        else if (     (MINEXPONENT(real_sp) /= -125) &
                 .OR. (MAXEXPONENT(real_sp) /=  128)) then
            ierr = MILHOJA_ERROR_BAD_FP32_EXPONENT
            RETURN
        end if

        !!!!!----- IEEE double precision
        ! Sanity check since we print info for this type
        if      (SIZEOF(real_dp) /= 8) then
            ierr = MILHOJA_ERROR_BAD_FP64_SIZE
            RETURN
        else if (EPSILON(real_dp) /= 2.220446049250313E-16_dp) then
            ierr = MILHOJA_ERROR_BAD_FP64_EPSILON
            RETURN
        else if (DIGITS(real_dp) /= 53) then
            ierr = MILHOJA_ERROR_BAD_FP64_DIGITS
            RETURN
        else if (     (MINEXPONENT(real_dp) /= -1021) &
                 .OR. (MAXEXPONENT(real_dp) /=  1024)) then
            ierr = MILHOJA_ERROR_BAD_FP64_EXPONENT
            RETURN
        end if

        !!!!!----- Confirm that mapped types are same
        if      (SIZEOF(real_default) /= SIZEOF(real_milhoja)) then
            ierr = MILHOJA_ERROR_REAL_SIZE_MISMATCH
            RETURN
        else if (EPSILON(real_default) /= EPSILON(real_milhoja)) then
            ierr = MILHOJA_ERROR_REAL_EPSILON_MISMATCH
            RETURN
        else if (DIGITS(real_default) /= DIGITS(real_milhoja)) then
            ierr = MILHOJA_ERROR_REAL_DIGITS_MISMATCH
            RETURN
        else if (     (MINEXPONENT(real_default) /= MINEXPONENT(real_milhoja)) &
                 .OR. (MAXEXPONENT(real_default) /= MAXEXPONENT(real_milhoja))) then
            ierr = MILHOJA_ERROR_REAL_EXPONENT_MISMATCH
            RETURN
        end if

        ierr = MILHOJA_SUCCESS
    end subroutine check_real_types

end module milhoja_types_mod

