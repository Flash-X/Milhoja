module roundtrip_kernel_mod
  use milhoja_types_mod, only: MILHOJA_REAL
  implicit none
  private
  public :: roundtrip_kernel

contains

  subroutine roundtrip_kernel(tile_lbound, field, scale, checksum)
    integer, intent(in) :: tile_lbound(:)
    real(MILHOJA_REAL), intent(inout) :: field(:, :, :, :)
    real(MILHOJA_REAL), intent(in) :: scale
    real(MILHOJA_REAL), intent(out) :: checksum(:)

    integer :: i, j, k, variable
    real(MILHOJA_REAL) :: increment, metadata_code

    metadata_code = real(sum(tile_lbound), MILHOJA_REAL)
    checksum(1) = 0.0_MILHOJA_REAL

#ifdef ROUNDTRIP_VARIABLE_FIRST
    do variable = 1, size(field, 1)
      do k = 1, size(field, 4)
        do j = 1, size(field, 3)
          do i = 1, size(field, 2)
            increment = real(100 * variable + 10 * j + i, MILHOJA_REAL) + &
                        metadata_code
            field(variable, i, j, k) = scale * field(variable, i, j, k) + &
                                        increment
            checksum(1) = checksum(1) + field(variable, i, j, k)
          end do
        end do
      end do
    end do
#else
    do variable = 1, size(field, 4)
      do k = 1, size(field, 3)
        do j = 1, size(field, 2)
          do i = 1, size(field, 1)
            increment = real(100 * variable + 10 * j + i, MILHOJA_REAL) + &
                        metadata_code
            field(i, j, k, variable) = scale * field(i, j, k, variable) + &
                                        increment
            checksum(1) = checksum(1) + field(i, j, k, variable)
          end do
        end do
      end do
    end do
#endif
  end subroutine roundtrip_kernel

end module roundtrip_kernel_mod
