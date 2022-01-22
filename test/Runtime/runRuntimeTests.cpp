#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>
#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Runtime.h>

#include "Base.h"
#include "setInitialConditions.h"
#include "errorEstBlank.h"
#include "Flash_par.h"

// It appears that OpenACC on Summit with PGI has max 32 asynchronous
// queues.  If you assign more CUDA streams to queues with OpenACC, then
// these streams just roll over and the last 32 CUDA streams will be the
// only streams mapped to queues.
constexpr int            N_STREAMS = 32; 
constexpr unsigned int   N_THREAD_TEAMS = 3;
constexpr unsigned int   N_THREADS_PER_TEAM = 10;
constexpr std::size_t    MEMORY_POOL_SIZE_BYTES = 4294967296; 

// We need to create our own main for the testsuite since we can only call
// MPI_Init/MPI_Finalize once per testsuite execution.
int main(int argc, char* argv[]) {
    using namespace milhoja;

    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    Logger::instantiate("RuntimeTest.log", GLOBAL_COMM, LEAD_RANK);
    Runtime::instantiate(N_THREAD_TEAMS, N_THREADS_PER_TEAM,
                         N_STREAMS, MEMORY_POOL_SIZE_BYTES);

    // Access config singleton within limited local scope so that it can't be
    // used by the rest of the application code outside the block.
    {
        GridConfiguration&   cfg = GridConfiguration::instance();

        cfg.xMin            = rp_Grid::X_MIN;
        cfg.xMax            = rp_Grid::X_MAX;
        cfg.yMin            = rp_Grid::Y_MIN;
        cfg.yMax            = rp_Grid::Y_MAX;
        cfg.zMin            = rp_Grid::Z_MIN;
        cfg.zMax            = rp_Grid::Z_MAX;
        cfg.nxb             = rp_Grid::NXB;
        cfg.nyb             = rp_Grid::NYB;
        cfg.nzb             = rp_Grid::NZB;
        cfg.nCcVars         = NUNKVAR;
        cfg.nGuard          = NGUARD;
        cfg.nBlocksX        = rp_Grid::N_BLOCKS_X;
        cfg.nBlocksY        = rp_Grid::N_BLOCKS_Y;
        cfg.nBlocksZ        = rp_Grid::N_BLOCKS_Z;
        cfg.maxFinestLevel  = rp_Grid::LREFINE_MAX;
        cfg.errorEstimation = Simulation::errorEstBlank;
        cfg.mpiComm         = GLOBAL_COMM;

        cfg.load();
    }
    Grid::initialize();

    RuntimeAction   initBlock_cpu;
    initBlock_cpu.name = "initBlock_cpu";
    initBlock_cpu.teamType        = ThreadTeamDataType::BLOCK;
    initBlock_cpu.nInitialThreads = rp_Simulation::N_THREADS_FOR_IC;
    initBlock_cpu.nTilesPerPacket = 0;
    initBlock_cpu.routine         = ActionRoutines::setInitialConditions_tile_cpu;

    Grid&   grid = Grid::instance();
    grid.initDomain(initBlock_cpu);

    int  rank = -1;
    MPI_Comm_rank(GLOBAL_COMM, &rank);

    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (rank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }

    int exitCode = RUN_ALL_TESTS();

    grid.destroyDomain();
    grid.finalize();

    MPI_Finalize();

    return exitCode;
}

