#include "Grid.h"
#include "RuntimeAction.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"

#include "Flash.h"
#include "constants.h"

#include "setInitialConditions.h"
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
        Grid::instance().initDomain(ActionRoutines::setInitialConditions_tile_cpu);
    }

    ~TestRuntime(void) {
        Grid::instance().destroyDomain();
    }

    void checkSolution(void) {
        RuntimeAction    computeError;
        computeError.name            = "ComputeErrors";
        computeError.nInitialThreads = 6;
        computeError.teamType        = ThreadTeamDataType::BLOCK;
        computeError.nTilesPerPacket = 0;
        computeError.routine         = ActionRoutines::computeErrors_tile_cpu;

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        Runtime::instance().executeCpuTasks("Analysis", computeError);

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
    Logger::instance().log("[googletest] Start TestCpuOnlyConfig");

    RuntimeAction    computeLaplacianDensity;
    RuntimeAction    computeLaplacianEnergy;

    computeLaplacianDensity.name            = "LaplacianDensity";
    computeLaplacianDensity.nInitialThreads = 6;
    computeLaplacianDensity.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianDensity.nTilesPerPacket = 0;
    computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

    computeLaplacianEnergy.name            = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 6;
    computeLaplacianEnergy.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianEnergy.nTilesPerPacket = 0;
    computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_tile_cpu;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuTasks("LapDens", computeLaplacianDensity);
    Runtime::instance().executeCpuTasks("LapEner", computeLaplacianEnergy);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End TestCpuOnlyConfig");
}

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestGpuOnlyConfig) {
    Logger::instance().log("[googletest] Start TestGpuOnlyConfig");

    RuntimeAction    computeLaplacianDensity;
    RuntimeAction    computeLaplacianEnergy;

    computeLaplacianDensity.name            = "LaplacianDensity";
    computeLaplacianDensity.nInitialThreads = 3;
    computeLaplacianDensity.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianDensity.nTilesPerPacket = 20;
    computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

    computeLaplacianEnergy.name            = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 3;
    computeLaplacianEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy.nTilesPerPacket = 20;
    computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("LapDens", computeLaplacianDensity);
    Runtime::instance().executeGpuTasks("LapEner", computeLaplacianEnergy);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End TestGpuOnlyConfig");
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestCpuGpuConfig) {
    Logger::instance().log("[googletest] Start TestCpuGpu");

    RuntimeAction    computeLaplacianDensity;
    RuntimeAction    computeLaplacianEnergy;

    computeLaplacianDensity.name            = "LaplacianDensity";
    computeLaplacianDensity.nInitialThreads = 3;
    computeLaplacianDensity.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianDensity.nTilesPerPacket = 0;
    computeLaplacianDensity.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

    computeLaplacianEnergy.name            = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 3;
    computeLaplacianEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy.nTilesPerPacket = 20;
    computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuTasks("ConcurrentCpuGpu",
                                           computeLaplacianDensity,
                                           computeLaplacianEnergy);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End TestCpuGpu");
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestSharedCpuGpuConfig) {
    Logger::instance().log("[googletest] Start Data Parallel Cpu/Gpu");

    RuntimeAction    computeLaplacian_cpu;
    RuntimeAction    computeLaplacian_gpu;

    computeLaplacian_cpu.name            = "LaplacianDensity_cpu";
    computeLaplacian_cpu.nInitialThreads = 4;
    computeLaplacian_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacian_cpu.nTilesPerPacket = 0;
    computeLaplacian_cpu.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

    computeLaplacian_gpu.name            = "LaplacianDensity_gpu";
    computeLaplacian_gpu.nInitialThreads = 2;
    computeLaplacian_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacian_gpu.nTilesPerPacket = 20;
    computeLaplacian_gpu.routine         = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuSplitTasks("DataParallelDensity",
                                                computeLaplacian_cpu,
                                                computeLaplacian_gpu,
                                                30);

    computeLaplacian_cpu.name    = "LaplacianEnergy_cpu";
    computeLaplacian_cpu.routine = ActionRoutines::computeLaplacianEnergy_tile_cpu;

    computeLaplacian_gpu.name    = "LaplacianEnergy_gpu";
    computeLaplacian_gpu.routine = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    Runtime::instance().executeCpuGpuSplitTasks("DataParallelEnergy",
                                                computeLaplacian_cpu,
                                                computeLaplacian_gpu,
                                                30);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Data Parallel Cpu/Gpu");
}
#endif

}

