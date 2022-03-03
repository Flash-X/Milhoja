#include "Milhoja.h"

!> A module in the Milhoja Fortran/C++ interoperability layer that declares the
!! Fortran types used in the layer and exposed to calling code.
!!
!! All Fortran routines in the Milhoja high-level Fortran interface shall use
!! the integer and real kinds defined in this module as opposed to the
!! C_* kinds in iso_c_binding.
!!
!! \todo The code will likely fail if MILHOJA_INT is *not* set to C_INT.  This
!!       is due to the fact that the C++ code is written with int/unsigned int
!!       rather than a dedicated type that can be easily changed.  Investigate
!!       the possibility of dedicated int/uint types in the C++ code.
module Milhoja_types_mod
    use IEEE_ARITHMETIC, ONLY : IEEE_SELECTED_REAL_KIND
    use iso_c_binding,   ONLY : C_INT, &
                                C_SIZE_T, &
                                C_DOUBLE

    implicit none
    private

    integer, parameter :: i32 = selected_int_kind(9)
    integer, parameter :: i64 = selected_int_kind(18)
    integer, parameter :: sp  = IEEE_SELECTED_REAL_KIND(p=6,  r=37)
    integer, parameter :: dp  = IEEE_SELECTED_REAL_KIND(p=15, r=307)

    !!!!!----- PUBLIC INTERFACE
    integer, parameter, public :: MILHOJA_INT    = C_INT
    integer, parameter, public :: MILHOJA_SIZE_T = C_SIZE_T
#ifdef MILHOJA_REAL_IS_DOUBLE
    integer, parameter, public :: MILHOJA_REAL   = C_DOUBLE
#else
#error "Invalid real type for Fortran interface"
#endif
end module Milhoja_types_mod

