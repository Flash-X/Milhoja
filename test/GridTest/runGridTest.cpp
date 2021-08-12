#include "Grid.h"
#include "OrchestrationLogger.h"
#include "Orchestration_constants.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
        ::testing::InitGoogleTest(&argc, argv);

        orchestration::grid_rp rp;
        rp.x_min       = 0.0_wp;
        rp.x_max       = 1.0_wp;
        rp.y_min       = 0.0_wp;
        rp.y_max       = 1.0_wp;
        rp.z_min       = 0.0_wp;
        rp.z_max       = 1.0_wp;
        rp.lrefine_max = 3;
        rp.nblockx     = 1+3*K1D;
        rp.nblocky     = 1+3*K2D;
        rp.nblockz     = 1+3*K3D;
        rp.nxb         = 1+7*K1D;
        rp.nyb         = 1+7*K2D;
        rp.nzb         = 1+7*K3D;

        orchestration::Logger::instantiate("GridUnitTest.log");

        orchestration::Grid::instantiate(rp);

        return RUN_ALL_TESTS();
}

