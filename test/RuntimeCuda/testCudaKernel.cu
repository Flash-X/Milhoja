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
    RuntimeAction    computeLaplacianDensity_packet;
    computeLaplacianDensity_packet.nInitialThreads = 6;
    computeLaplacianDensity_packet.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianDensity_packet.nTilesPerPacket = 1;
    computeLaplacianDensity_packet.routine = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

    RuntimeAction    computeLaplacianEnergy_packet;
    computeLaplacianEnergy_packet.nInitialThreads = 6;
    computeLaplacianEnergy_packet.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy_packet.nTilesPerPacket = 1;
    computeLaplacianEnergy_packet.routine = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    CudaRuntime::instance().executeGpuTasks("Density", computeLaplacianDensity_packet);
    CudaRuntime::instance().executeGpuTasks("Energy",  computeLaplacianEnergy_packet);

    // TODO: Where to get this value from?  It should be a runtime parameter.
    constexpr Real   ENERGY_SCALE_FACTOR = 5.0;
    std::unique_ptr<Tile>    tileDesc{};
    for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next()) {
        tileDesc = ti->buildCurrentTile();

        const int        level   = tileDesc->level();
        const IntVect    lo      = tileDesc->lo();
        const IntVect    hi      = tileDesc->hi();
        FArray4D         f       = tileDesc->data();
        const FArray1D   xCoords = grid.getCellCoords(Axis::I, Edge::Center,
                                                      level, lo, hi); 
        const FArray1D   yCoords = grid.getCellCoords(Axis::J, Edge::Center,
                                                      level, lo, hi); 
       
        StaticPhysicsRoutines::scaleEnergy(lo, hi, xCoords, yCoords, f,
                                           ENERGY_SCALE_FACTOR);
    }

    //***** ANALYSIS RUNTIME EXECUTION CYCLE
    RuntimeAction    computeError_block;
    computeError_block.nInitialThreads     = 6;
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

