#include "Grid.h"
#include "RuntimeAction.h"
#include "CudaRuntime.h"

#include "Flash.h"
#include "constants.h"

#include "setInitialConditions_block.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"
#include "scaleEnergy.h"
#include "Analysis.h"

#include "gtest/gtest.h"

using namespace orchestration;

namespace {

class TestRuntime : public testing::Test {
protected:
    TestRuntime(void) {
        Grid::instance().initDomain(Simulation::setInitialConditions_block);
    }

    ~TestRuntime(void) {
        Grid::instance().destroyDomain();
    }

    void checkSolution(void) {
        RuntimeAction    computeError_block;
        computeError_block.name                = "ComputeErrors";
        computeError_block.nInitialThreads     = 6;
        computeError_block.teamType            = ThreadTeamDataType::BLOCK;
        computeError_block.nTilesPerPacket     = 0;
        computeError_block.routine             = Analysis::computeErrors_block;

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        CudaRuntime::instance().executeCpuTasks("Analysis", computeError_block);

        double L_inf1      = 0.0;
        double meanAbsErr1 = 0.0;
        double L_inf2      = 0.0;
        double meanAbsErr2 = 0.0;
        Analysis::densityErrors(&L_inf1, &meanAbsErr1);
        Analysis::energyErrors(&L_inf2, &meanAbsErr2);
        std::cout << "L_inf1 = " << L_inf1 << "\n";
        std::cout << "L_inf2 = " << L_inf2 << std::endl;

        EXPECT_TRUE(0.0 <= L_inf1);
        EXPECT_TRUE(L_inf1 <= 1.0e-15);
        EXPECT_TRUE(0.0 <= meanAbsErr1);
        EXPECT_TRUE(meanAbsErr1 <= 1.0e-15);

        EXPECT_TRUE(0.0 <= L_inf2);
        EXPECT_TRUE(L_inf2 <= 9.0e-6);
        EXPECT_TRUE(0.0 <= meanAbsErr2);
        EXPECT_TRUE(meanAbsErr2 <= 9.0e-6);
    }
};

TEST_F(TestRuntime, TestCpuOnlyConfig) {
    RuntimeAction    computeLaplacianDensity;
    computeLaplacianDensity.name = "LaplacianDensity";
    computeLaplacianDensity.nInitialThreads = 6;
    computeLaplacianDensity.teamType = ThreadTeamDataType::BLOCK;
    computeLaplacianDensity.nTilesPerPacket = 0;
    computeLaplacianDensity.routine = ActionRoutines::computeLaplacianDensity_tile_cpu;

    RuntimeAction    computeLaplacianEnergy;
    computeLaplacianEnergy.name = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 6;
    computeLaplacianEnergy.teamType = ThreadTeamDataType::BLOCK;
    computeLaplacianEnergy.nTilesPerPacket = 0;
    computeLaplacianEnergy.routine = ActionRoutines::computeLaplacianEnergy_tile_cpu;

    RuntimeAction    scaleEnergy;
    scaleEnergy.name = "scaleEnergy";
    scaleEnergy.nInitialThreads = 6;
    scaleEnergy.teamType = ThreadTeamDataType::BLOCK;
    scaleEnergy.nTilesPerPacket = 0;
    scaleEnergy.routine = ActionRoutines::scaleEnergy_tile_cpu;

    CudaRuntime::instance().executeCpuTasks("LapDens", computeLaplacianDensity);
    CudaRuntime::instance().executeCpuTasks("LapEner", computeLaplacianEnergy);
    CudaRuntime::instance().executeCpuTasks("scEner",  scaleEnergy);

    checkSolution();
}

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestGpuOnlyConfig) {
    RuntimeAction    computeLaplacianDensity;
    computeLaplacianDensity.name = "LaplacianDensity";
    computeLaplacianDensity.nInitialThreads = 6;
    computeLaplacianDensity.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianDensity.nTilesPerPacket = 1;
    computeLaplacianDensity.routine = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

    RuntimeAction    computeLaplacianEnergy;
    computeLaplacianEnergy.name = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 6;
    computeLaplacianEnergy.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy.nTilesPerPacket = 1;
    computeLaplacianEnergy.routine = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    RuntimeAction    scaleEnergy;
    scaleEnergy.name = "scaleEnergy";
    scaleEnergy.nInitialThreads = 6;
    scaleEnergy.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    scaleEnergy.nTilesPerPacket = 1;
    scaleEnergy.routine = ActionRoutines::scaleEnergy_packet_oacc_summit;

    CudaRuntime::instance().executeGpuTasks("LapDens", computeLaplacianDensity);
    CudaRuntime::instance().executeGpuTasks("LapEner", computeLaplacianEnergy);
    CudaRuntime::instance().executeGpuTasks("scEner",  scaleEnergy);

    checkSolution();
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestFullConfig) {
    RuntimeAction    computeLaplacianDensity;
    computeLaplacianDensity.name = "LaplacianDensity";
    computeLaplacianDensity.nInitialThreads = 2;
    computeLaplacianDensity.teamType = ThreadTeamDataType::BLOCK;
    computeLaplacianDensity.nTilesPerPacket = 0;
    computeLaplacianDensity.routine = ActionRoutines::computeLaplacianDensity_tile_cpu;

    RuntimeAction    computeLaplacianEnergy;
    computeLaplacianEnergy.name = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 5;
    computeLaplacianEnergy.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy.nTilesPerPacket = 1;
    computeLaplacianEnergy.routine = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    RuntimeAction    scaleEnergy;
    scaleEnergy.name = "scaleEnergy";
    scaleEnergy.nInitialThreads = 0;
    scaleEnergy.teamType = ThreadTeamDataType::BLOCK;
    scaleEnergy.nTilesPerPacket = 0;
    scaleEnergy.routine = ActionRoutines::scaleEnergy_tile_cpu;

    CudaRuntime::instance().executeTasks_FullPacket("FullPacket",
                                                    computeLaplacianDensity,
                                                    computeLaplacianEnergy,
                                                    scaleEnergy);

    checkSolution();
}
#endif

}

