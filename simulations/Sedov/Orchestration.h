#ifndef ORCHESTRATION_H__
#define ORCHESTRATION_H__

namespace orch {
    // Fix runtime parameters - Setup for Summit with 7 cores/MPI process
    // and 1 GPU/MPI process
    constexpr unsigned int   nThreadTeams        = 1;
    constexpr unsigned int   nThreadsPerTeam     = 7;
    constexpr int            nStreams            = 32; 
    constexpr std::size_t    memoryPoolSizeBytes = 12884901888;
};

#endif

