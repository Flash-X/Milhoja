#include <gtest/gtest.h>

#include <mpi.h>

#include "OrchestrationLogger.h"
#include "GridConfiguration.h"
#include "Grid.h"

#include "constants.h"
#include "Flash.h"
#include "Flash_par.h"

int main(int argc, char* argv[]) {
    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    orchestration::Logger::instantiate("GridUnitTest.log",
                                       GLOBAL_COMM, LEAD_RANK);

    // Access config singleton within limited local scope so that it can't be
    // used by the rest of the application code outside the block.
    {
        orchestration::GridConfiguration&   cfg = orchestration::GridConfiguration::instance();
        cfg.xMin           = rp_Grid::X_MIN;
        cfg.xMax           = rp_Grid::X_MAX;
        cfg.yMin           = rp_Grid::Y_MIN;
        cfg.yMax           = rp_Grid::Y_MAX;
        cfg.zMin           = rp_Grid::Z_MIN;
        cfg.zMax           = rp_Grid::Z_MAX;
        cfg.nxb            = NXB;
        cfg.nyb            = NYB;
        cfg.nzb            = NZB;
        cfg.nCcVars        = NUNKVAR;
        cfg.nGuard         = NGUARD;
        cfg.nBlocksX       = rp_Grid::N_BLOCKS_X;
        cfg.nBlocksY       = rp_Grid::N_BLOCKS_Y;
        cfg.nBlocksZ       = rp_Grid::N_BLOCKS_Z;
        cfg.maxFinestLevel = rp_Grid::LREFINE_MAX;

        orchestration::Grid::instantiate();
    }

    return RUN_ALL_TESTS();
}

