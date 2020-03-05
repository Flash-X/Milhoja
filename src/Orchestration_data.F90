!<
!!
!!  

module Orchestration_data
    implicit none

    ! Keep track of the initialization status of the OrchestrationRuntime
    ! singleton so that we only instantiate this class once per simulation
    logical, save :: isRuntimeInitialized = .FALSE.
end module Orchestration_data

