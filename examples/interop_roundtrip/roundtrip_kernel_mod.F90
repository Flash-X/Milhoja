#include "Milhoja.h"

module roundtrip_kernel_mod
  use milhoja_types_mod, only: MILHOJA_REAL
  implicit none
  private
  public :: roundtrip_kernel

#ifdef MILHOJA_OPENMP_OFFLOADING
  !$omp declare target(roundtrip_kernel)
#endif
contains

  subroutine roundtrip_kernel(tile_lbound, field, scale, checksum)
    !$acc routine vector
    integer, intent(in) :: tile_lbound(:)
    real(MILHOJA_REAL), intent(inout) :: field(:, :, :, :)
    real(MILHOJA_REAL), intent(in) :: scale
    real(MILHOJA_REAL), intent(out) :: checksum(:)

    integer :: i, j, k, variable
    real(MILHOJA_REAL) :: checksum_value, increment, metadata_code

    metadata_code = real(sum(tile_lbound), MILHOJA_REAL)
    checksum_value = 0.0_MILHOJA_REAL

#ifdef ROUNDTRIP_VARIABLE_FIRST
    !$acc loop vector collapse(4) reduction(+:checksum_value)
    do variable = 1, size(field, 1)
      do k = 1, size(field, 4)
        do j = 1, size(field, 3)
          do i = 1, size(field, 2)
            increment = real(100 * variable + 10 * j + i, MILHOJA_REAL) + &
                        metadata_code
            field(variable, i, j, k) = scale * field(variable, i, j, k) + &
                                        increment
            checksum_value = checksum_value + field(variable, i, j, k)
          end do
        end do
      end do
    end do
#else
    !$acc loop vector collapse(4) reduction(+:checksum_value)
    do variable = 1, size(field, 4)
      do k = 1, size(field, 3)
        do j = 1, size(field, 2)
          do i = 1, size(field, 1)
            increment = real(100 * variable + 10 * j + i, MILHOJA_REAL) + &
                        metadata_code
            field(i, j, k, variable) = scale * field(i, j, k, variable) + &
                                        increment
            checksum_value = checksum_value + field(i, j, k, variable)
          end do
        end do
      end do
    end do
#endif
    checksum(1) = checksum_value
    ! Return the device-only scratch result through the in/out field as well.
    ! This lets the host driver validate scratch without dereferencing device
    ! memory.  In either layout these indices select logical (1,1,1,var=1).
    field(1, 1, 1, 1) = checksum_value
  end subroutine roundtrip_kernel

end module roundtrip_kernel_mod
