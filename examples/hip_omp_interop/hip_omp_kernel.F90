module hip_omp_kernel_mod
  use, intrinsic :: iso_c_binding, only : c_double, c_int
  use omp_lib, only : omp_is_initial_device
  implicit none
  private

  public :: hip_omp_transform_variable_last
  public :: hip_omp_transform_variable_first

contains

  subroutine hip_omp_transform_variable_last(values, n_x, n_y, n_z, &
      n_variables, ran_on_device) bind(C, &
      name="hip_omp_transform_variable_last")
    integer(c_int), value, intent(in) :: n_x
    integer(c_int), value, intent(in) :: n_y
    integer(c_int), value, intent(in) :: n_z
    integer(c_int), value, intent(in) :: n_variables
    real(c_double), intent(inout) :: values(n_x, n_y, n_z, n_variables)
    integer(c_int), intent(out) :: ran_on_device

    integer :: i, j, k, variable
    real(c_double) :: logical_index

    ran_on_device = 0_c_int

    !$omp target teams distribute parallel do is_device_ptr(values) &
    !$omp& map(from: ran_on_device) collapse(4) private(logical_index)
    do variable = 1, n_variables
      do k = 1, n_z
        do j = 1, n_y
          do i = 1, n_x
            logical_index = real(i - 1, c_double) &
                + 10.0_c_double * real(j - 1, c_double) &
                + 100.0_c_double * real(k - 1, c_double) &
                + 1000.0_c_double * real(variable - 1, c_double)
            values(i, j, k, variable) = &
                1.5_c_double * values(i, j, k, variable) + logical_index
            if (i == 1 .and. j == 1 .and. k == 1 .and. variable == 1) then
              if (.not. omp_is_initial_device()) ran_on_device = 1_c_int
            end if
          end do
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine hip_omp_transform_variable_last

  subroutine hip_omp_transform_variable_first(values, n_x, n_y, n_z, &
      n_variables, ran_on_device) bind(C, &
      name="hip_omp_transform_variable_first")
    integer(c_int), value, intent(in) :: n_x
    integer(c_int), value, intent(in) :: n_y
    integer(c_int), value, intent(in) :: n_z
    integer(c_int), value, intent(in) :: n_variables
    real(c_double), intent(inout) :: values(n_variables, n_x, n_y, n_z)
    integer(c_int), intent(out) :: ran_on_device

    integer :: i, j, k, variable
    real(c_double) :: logical_index

    ran_on_device = 0_c_int

    !$omp target teams distribute parallel do is_device_ptr(values) &
    !$omp& map(from: ran_on_device) collapse(4) private(logical_index)
    do k = 1, n_z
      do j = 1, n_y
        do i = 1, n_x
          do variable = 1, n_variables
            logical_index = real(i - 1, c_double) &
                + 10.0_c_double * real(j - 1, c_double) &
                + 100.0_c_double * real(k - 1, c_double) &
                + 1000.0_c_double * real(variable - 1, c_double)
            values(variable, i, j, k) = &
                1.5_c_double * values(variable, i, j, k) + logical_index
            if (i == 1 .and. j == 1 .and. k == 1 .and. variable == 1) then
              if (.not. omp_is_initial_device()) ran_on_device = 1_c_int
            end if
          end do
        end do
      end do
    end do
    !$omp end target teams distribute parallel do
  end subroutine hip_omp_transform_variable_first

end module hip_omp_kernel_mod

