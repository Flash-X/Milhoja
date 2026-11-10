module hip_omp_kernel_mod
  use, intrinsic :: iso_c_binding, only : c_double, c_int
  use omp_lib, only : omp_is_initial_device
  implicit none
  private

  public :: hip_omp_transform

contains

  subroutine hip_omp_transform(values, n_values, ran_on_device) &
      bind(C, name="hip_omp_transform")
    integer(c_int), value, intent(in) :: n_values
    real(c_double), intent(inout) :: values(*)
    integer(c_int), intent(out) :: ran_on_device

    integer :: i

    ran_on_device = 0_c_int

    !$omp target teams distribute parallel do is_device_ptr(values) &
    !$omp& map(from: ran_on_device)
    do i = 1, n_values
      values(i) = 2.0_c_double * values(i) + 1.0_c_double
      if (i == 1) then
        if (.not. omp_is_initial_device()) ran_on_device = 1_c_int
      end if
    end do
    !$omp end target teams distribute parallel do
  end subroutine hip_omp_transform

end module hip_omp_kernel_mod

