!<
!!
!!  

module Analysis_data
    implicit none

    real, save, allocatable :: an_LinfErrors(:, :)
    real, save, allocatable :: an_meanAbsErrors(:, :)

    real, save :: an_energyFactor = 0.0
end module Analysis_data

