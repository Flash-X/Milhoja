!<
!!
!!  

module Orchestration_data
    implicit none

    character(18), parameter :: or_logFilename        = "TestRuntimeF90.log"
    integer,       parameter :: or_nTileThreadTeams   = 3
    integer,       parameter :: or_nPacketThreadTeams = 0
    integer,       parameter :: or_nThreadsPerTeam    = 5

    ! Keep track of the initialization status of the OrchestrationRuntime
    ! singleton so that we only instantiate this class once per simulation
    logical,       save      :: or_isRuntimeInitialized = .FALSE.
end module Orchestration_data

