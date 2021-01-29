#include "Grid.h"
#include "RuntimeAction.h"
#include "Runtime.h"
#include "OrchestrationLogger.h"
#include "errorEstBlank.h"

#include "Flash.h"
#include "Flash_par.h"
#include "constants.h"

#include "setInitialConditions.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"
#include "computeLaplacianFused.h"
#include "Analysis.h"

#include "gtest/gtest.h"

using namespace orchestration;

namespace {

class TestRuntime : public testing::Test {
protected:
    TestRuntime(void) {
        Grid::instance().initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                                    rp_Simulation::N_THREADS_FOR_IC,
                                    Simulation::errorEstBlank);
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

        Analysis::initialize(  rp_Grid::N_BLOCKS_X
                             * rp_Grid::N_BLOCKS_Y
                             * rp_Grid::N_BLOCKS_Z);
        Runtime::instance().executeCpuTasks("Analysis",
                                            rp_Simulation::N_DISTRIBUTOR_THREADS,
                                            computeError);

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
    Runtime::instance().executeCpuTasks("LapDens", 
                                        rp_Simulation::N_DISTRIBUTOR_THREADS,
                                        computeLaplacianDensity);
    Runtime::instance().executeCpuTasks("LapEner",
                                        rp_Simulation::N_DISTRIBUTOR_THREADS,
                                        computeLaplacianEnergy);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End TestCpuOnlyConfig");
}

TEST_F(TestRuntime, TestFusedKernelsCpu) {
    Logger::instance().log("[googletest] Start Fused Kernels - Host");

    RuntimeAction    computeLaplacianFused_cpu;

    computeLaplacianFused_cpu.name            = "LaplacianFusedKernels_cpu";
    computeLaplacianFused_cpu.nInitialThreads = 6;
    computeLaplacianFused_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianFused_cpu.nTilesPerPacket = 0;
    computeLaplacianFused_cpu.routine         = ActionRoutines::computeLaplacianFusedKernels_tile_cpu;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuTasks("Fused Kernels CPU",
                                        rp_Simulation::N_DISTRIBUTOR_THREADS,
                                        computeLaplacianFused_cpu);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Kernels - Host");
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

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestSharedCpuGpuWowza) {
    Logger::instance().log("[googletest] Start Cpu/Gpu Wowza Config");

    RuntimeAction    computeLaplacianDensity_cpu;
    RuntimeAction    computeLaplacianDensity_gpu;
    RuntimeAction    computeLaplacianEnergy_gpu;

    computeLaplacianDensity_cpu.name            = "LaplacianDensity_cpu";
    computeLaplacianDensity_cpu.nInitialThreads = 2;
    computeLaplacianDensity_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianDensity_cpu.nTilesPerPacket = 0;
    computeLaplacianDensity_cpu.routine         = ActionRoutines::computeLaplacianDensity_tile_cpu;

    computeLaplacianDensity_gpu.name            = "LaplacianDensity_gpu";
    computeLaplacianDensity_gpu.nInitialThreads = 3;
    computeLaplacianDensity_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianDensity_gpu.nTilesPerPacket = 20;
    computeLaplacianDensity_gpu.routine         = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

    computeLaplacianEnergy_gpu.name            = "LaplacianEnergy_gpu";
    computeLaplacianEnergy_gpu.nInitialThreads = 3;
    computeLaplacianEnergy_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy_gpu.nTilesPerPacket = 20;
    computeLaplacianEnergy_gpu.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuWowzaTasks("CPU/GPU Wowza",
                                                computeLaplacianDensity_cpu,
                                                computeLaplacianDensity_gpu,
                                                computeLaplacianEnergy_gpu,
                                                20);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Cpu/Gpu Wowza Config");
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestFusedActions) {
    Logger::instance().log("[googletest] Start Fused Actions");

    RuntimeAction    computeLaplacianFused_gpu;

    computeLaplacianFused_gpu.name            = "LaplacianFusedActions_gpu";
    computeLaplacianFused_gpu.nInitialThreads = 3;
    computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianFused_gpu.nTilesPerPacket = 20;
    computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("Fused Actions GPU", computeLaplacianFused_gpu);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Actions");
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestFusedKernelsStrong) {
    Logger::instance().log("[googletest] Start Fused Kernels Strong");

    RuntimeAction    computeLaplacianFused_gpu;

    computeLaplacianFused_gpu.name            = "LaplacianFusedKernelsStrong_gpu";
    computeLaplacianFused_gpu.nInitialThreads = 3;
    computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianFused_gpu.nTilesPerPacket = 20;
    computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsStrong_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("Fused Kernels Strong GPU", computeLaplacianFused_gpu);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Kernels Strong");
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestFusedKernelsWeak) {
    Logger::instance().log("[googletest] Start Fused Kernels Weak");

    RuntimeAction    computeLaplacianFused_gpu;

    computeLaplacianFused_gpu.name            = "LaplacianFusedKernelsWeak_gpu";
    computeLaplacianFused_gpu.nInitialThreads = 3;
    computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianFused_gpu.nTilesPerPacket = 20;
    computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsWeak_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("Fused Kernels Weak GPU", computeLaplacianFused_gpu);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Kernels Weak");
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestSharedCpuGpuConfigFusedActions) {
    Logger::instance().log("[googletest] Start Data Parallel Cpu/Gpu Fused Actions");

    RuntimeAction    computeLaplacian_cpu;
    RuntimeAction    computeLaplacian_gpu;

    computeLaplacian_cpu.name            = "LaplacianFused_cpu";
    computeLaplacian_cpu.nInitialThreads = 3;
    computeLaplacian_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacian_cpu.nTilesPerPacket = 0;
    computeLaplacian_cpu.routine         = ActionRoutines::computeLaplacianFusedKernels_tile_cpu;

    computeLaplacian_gpu.name            = "LaplacianFused_gpu";
    computeLaplacian_gpu.nInitialThreads = 3;
    computeLaplacian_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacian_gpu.nTilesPerPacket = 20;
    computeLaplacian_gpu.routine         = ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuSplitTasks("DataParallelFused",
                                                computeLaplacian_cpu,
                                                computeLaplacian_gpu,
                                                15);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Data Parallel Cpu/Gpu Fused Actions");
}
#endif

#if defined(USE_CUDA_BACKEND)
TEST_F(TestRuntime, TestSharedCpuGpuConfigFusedKernels) {
    Logger::instance().log("[googletest] Start Data Parallel Cpu/Gpu Fused Kernels");

    RuntimeAction    computeLaplacian_cpu;
    RuntimeAction    computeLaplacian_gpu;

    computeLaplacian_cpu.name            = "LaplacianFused_cpu";
    computeLaplacian_cpu.nInitialThreads = 4;
    computeLaplacian_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacian_cpu.nTilesPerPacket = 0;
    computeLaplacian_cpu.routine         = ActionRoutines::computeLaplacianFusedKernels_tile_cpu;

    computeLaplacian_gpu.name            = "LaplacianFused_gpu";
    computeLaplacian_gpu.nInitialThreads = 2;
    computeLaplacian_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacian_gpu.nTilesPerPacket = 20;
    computeLaplacian_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsStrong_packet_oacc_summit;

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuSplitTasks("DataParallelFused",
                                                computeLaplacian_cpu,
                                                computeLaplacian_gpu,
                                                15);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Data Parallel Cpu/Gpu Fused Kernels");
}
#endif

}

