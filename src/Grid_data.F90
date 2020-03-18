!<
!!
!!  

#include "constants.h"

module Grid_data
    implicit none

    ! Keep track of the initialization status of the Grid
    ! singleton so that we only instantiate this class once per simulation
    logical, save :: gr_isGridInitialized = .FALSE.

    real,    save :: gr_globalDomain(LOW:HIGH, 1:MDIM)
end module Grid_data

