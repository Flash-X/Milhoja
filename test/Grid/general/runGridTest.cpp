#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>
#include <Milhoja_GridConfiguration.h>
#include <Milhoja_Grid.h>

#include "Base.h"
#include "setInitialConditions.h"
#include "errorEstMaximal.h"
#include "Flash_par.h"

int main(int argc, char* argv[]) {
    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    milhoja::Logger::instantiate("GridGeneralUnitTest.log",
                                 GLOBAL_COMM, LEAD_RANK);

    // Access config singleton within limited local scope so that it can't be
    // used by the rest of the application code outside the block.
    {
        milhoja::GridConfiguration&   cfg = milhoja::GridConfiguration::instance();

        cfg.xMin                     = rp_Grid::X_MIN;
        cfg.xMax                     = rp_Grid::X_MAX;
        cfg.yMin                     = rp_Grid::Y_MIN;
        cfg.yMax                     = rp_Grid::Y_MAX;
        cfg.zMin                     = rp_Grid::Z_MIN;
        cfg.zMax                     = rp_Grid::Z_MAX;
        cfg.nxb                      = rp_Grid::NXB;
        cfg.nyb                      = rp_Grid::NYB;
        cfg.nzb                      = rp_Grid::NZB;
        cfg.nCcVars                  = NUNKVAR;
        cfg.nGuard                   = NGUARD;
        cfg.nBlocksX                 = rp_Grid::N_BLOCKS_X;
        cfg.nBlocksY                 = rp_Grid::N_BLOCKS_Y;
        cfg.nBlocksZ                 = rp_Grid::N_BLOCKS_Z;
        cfg.maxFinestLevel           = rp_Grid::LREFINE_MAX;
        cfg.initBlock                = ActionRoutines::setInitialConditions_tile_cpu;
        cfg.nDistributorThreads_init = rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC;
        cfg.nCpuThreads_init         = rp_Simulation::N_THREADS_FOR_IC;
        cfg.errorEstimation          = Simulation::errorEstMaximal;

        cfg.load();

        milhoja::Grid::instantiate();
    }

    milhoja::Grid::instance().initDomain();

    int exitCode = RUN_ALL_TESTS();

    milhoja::Grid::instance().destroyDomain();

    return exitCode;
}

