!<
!!
!!  

module Orchestration_data
    use iso_c_binding, ONLY : C_INT, C_LONG_LONG

    implicit none

    character(18),        parameter :: or_logFilename         = "TestRuntimeF90.log"
    integer(C_INT),       parameter :: or_nThreadTeams        = 3
    integer(C_INT),       parameter :: or_nThreadsPerTeam     = 10
    integer(C_INT),       parameter :: or_nStreams            = 32
    integer(C_LONG_LONG), parameter :: or_nBytesInMemoryPools = 8589934592_C_LONG_LONG

    ! Keep track of the initialization status of the OrchestrationRuntime
    ! singleton so that we only instantiate this class once per simulation
    logical,       save      :: or_isRuntimeInitialized = .FALSE.
end module Orchestration_data

