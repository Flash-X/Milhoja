#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "Grid.h"
#include "RuntimeAction.h"
#include "CudaRuntime.h"
#include "CudaStreamManager.h"
#include "CudaMemoryManager.h"

#include "Flash.h"
#include "constants.h"

#include "setInitialConditions_block.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"
#include "scaleEnergy.h"
#include "Analysis.h"

int   main(int argc, char* argv[]) {
    // It appears that OpenACC on Summit with PGI has max 32 asynchronous
    // queues.  If you assign more CUDA streams to queues with OpenACC, then
    // these streams just roll over and the last 32 CUDA streams will be the
    // only streams mapped to queues.
    constexpr int            N_STREAMS = 32; 
    constexpr unsigned int   N_THREAD_TEAMS = 1;
    constexpr unsigned int   MAX_THREADS = 6;
    constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 4294967296; 
    constexpr std::size_t    N_BLOCKS = N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z;

    using namespace orchestration;

    CudaRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    CudaRuntime::setMaxThreadsPerTeam(MAX_THREADS);
    CudaRuntime::setLogFilename("DeleteMe.log");
    std::cout << "\n";
    std::cout << "----------------------------------------------------------\n";
    CudaRuntime::instance().printGpuInformation();
    std::cout << "----------------------------------------------------------\n";
    std::cout << std::endl;

    CudaStreamManager::setMaxNumberStreams(N_STREAMS);
    CudaMemoryManager::setBufferSize(MEMORY_POOL_SIZE_BYTES);

    //***** SET INITIAL CONDITIONS
    // Initialize Grid unit/AMReX
    Grid::instantiate();
    Grid&    grid = Grid::instance();
    grid.initDomain(Simulation::setInitialConditions_block);

    //***** FIRST RUNTIME EXECUTION CYCLE
    RuntimeAction    computeLaplacianDensity;
    computeLaplacianDensity.nInitialThreads = 6;
    computeLaplacianDensity.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianDensity.nTilesPerPacket = 1;
    computeLaplacianDensity.routine = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

    RuntimeAction    computeLaplacianEnergy;
    computeLaplacianEnergy.nInitialThreads = 6;
    computeLaplacianEnergy.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy.nTilesPerPacket = 1;
    computeLaplacianEnergy.routine = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    RuntimeAction    scaleEnergy;
    scaleEnergy.nInitialThreads = 6;
    scaleEnergy.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    scaleEnergy.nTilesPerPacket = 1;
    scaleEnergy.routine = ActionRoutines::scaleEnergy_packet_oacc_summit;

    CudaRuntime::instance().executeGpuTasks("Density", computeLaplacianDensity);
    CudaRuntime::instance().executeGpuTasks("Energy",  computeLaplacianEnergy);
    CudaRuntime::instance().executeGpuTasks("Scale",   scaleEnergy);

    //***** ANALYSIS RUNTIME EXECUTION CYCLE
    RuntimeAction    computeError_block;
    computeError_block.nInitialThreads     = 1;
    computeError_block.teamType            = ThreadTeamDataType::BLOCK;
    computeError_block.nTilesPerPacket     = 0;
    computeError_block.routine             = Analysis::computeErrors_block;

    Analysis::initialize(N_BLOCKS);
    CudaRuntime::instance().executeCpuTasks("Analysis", computeError_block);

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    Analysis::densityErrors(&L_inf1, &meanAbsErr1);
    Analysis::energyErrors(&L_inf2, &meanAbsErr2);
    std::cout << "L_inf1 = " << L_inf1 << "\n";
    std::cout << "L_inf2 = " << L_inf2 << std::endl;

//    EXPECT_TRUE(0.0 <= L_inf1);
//    EXPECT_TRUE(L_inf1 <= 1.0e-15);
//    EXPECT_TRUE(0.0 <= meanAbsErr1);
//    EXPECT_TRUE(meanAbsErr1 <= 1.0e-15);

//    EXPECT_TRUE(0.0 <= L_inf2);
//    EXPECT_TRUE(L_inf2 <= 9.0e-6);
//    EXPECT_TRUE(0.0 <= meanAbsErr2);
//    EXPECT_TRUE(meanAbsErr2 <= 9.0e-6);

    // Clean-up
    grid.destroyDomain();
}

