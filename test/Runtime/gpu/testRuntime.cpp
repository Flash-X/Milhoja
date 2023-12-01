#include <gtest/gtest.h>

#include <Milhoja_Logger.h>
#include <Milhoja_Grid.h>
#include <Milhoja_axis.h>
#include <Milhoja_edge.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Tile.h>
#include <Milhoja_RuntimeAction.h>
#include <Milhoja_Runtime.h>

#include "RuntimeParameters.h"
#include "setInitialConditions.h"
#include "computeLaplacianDensity.h"
#include "computeLaplacianEnergy.h"
#include "computeLaplacianFused.h"
#include "Analysis.h"

#include "cpu_tf_dens.h"
#include "Tile_cpu_tf_dens.h"
#include "cpu_tf_ener.h"
#include "Tile_cpu_tf_ener.h"
#include "cpu_tf_fused.h"
#include "Tile_cpu_tf_fused.h"
#include "cpu_tf_analysis.h"
#include "Tile_cpu_tf_analysis.h"

#include "DataPacket_gpu_tf_dens.h"
#include "DataPacket_gpu_tf_ener.h"
#include "DataPacket_gpu_tf_fused_actions.h"
#include "DataPacket_gpu_tf_fused_kernels.h"

using namespace milhoja;

namespace {

class TestRuntime : public testing::Test {
protected:
    TestRuntime(void) {
        // Each test can use the Grid structure determined by initDomain when
        // this application is started.  However, each test can overwrite the
        // ICs during execution.  Therefore, we blindly reset the ICs each time.
        Grid&    grid = Grid::instance();
        for (int level = 0; level<=grid.getMaxLevel(); ++level) {
            for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
                std::unique_ptr<Tile> tileDesc = ti->buildCurrentTile();

                const IntVect       loGC  = tileDesc->loGC();
                const IntVect       hiGC  = tileDesc->hiGC();
                FArray4D            U     = tileDesc->data();

                FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, level,
                                                      loGC, hiGC); 
                FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, level,
                                                      loGC, hiGC); 

                StaticPhysicsRoutines::setInitialConditions(loGC, hiGC,
                                                            xCoords, yCoords,
                                                            U);
            }
        }
    }

    ~TestRuntime(void) { }

    void checkSolution(void) {
        RuntimeAction    computeError;
        computeError.name            = "ComputeErrors";
//        computeError.nInitialThreads = 6;
        computeError.nInitialThreads = 1;
        computeError.teamType        = ThreadTeamDataType::BLOCK;
        computeError.nTilesPerPacket = 0;
        computeError.routine         = cpu_tf_analysis::taskFunction;

        RuntimeParameters&   RPs = RuntimeParameters::instance();

        unsigned int    nBlocksX{RPs.getUnsignedInt("Grid", "nBlocksX")};
        unsigned int    nBlocksY{RPs.getUnsignedInt("Grid", "nBlocksY")};
        unsigned int    nBlocksZ{RPs.getUnsignedInt("Grid", "nBlocksZ")};
        Analysis::initialize( nBlocksX * nBlocksY * nBlocksZ );
        Tile_cpu_tf_analysis::acquireScratch();
        const Tile_cpu_tf_analysis   prototype{};
        Runtime::instance().executeCpuTasks("Analysis",
                                            computeError, prototype);
        Tile_cpu_tf_analysis::releaseScratch();

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
    computeLaplacianDensity.routine         = cpu_tf_dens::taskFunction;

    computeLaplacianEnergy.name            = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 6;
    computeLaplacianEnergy.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianEnergy.nTilesPerPacket = 0;
    computeLaplacianEnergy.routine         = cpu_tf_ener::taskFunction;

    Tile_cpu_tf_dens::acquireScratch();
    Tile_cpu_tf_ener::acquireScratch();

    const Tile_cpu_tf_dens    prototypeDens{};
    const Tile_cpu_tf_ener    prototypeEner{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuTasks("LapDens",
                                        computeLaplacianDensity, prototypeDens);
    Runtime::instance().executeCpuTasks("LapEner",
                                        computeLaplacianEnergy, prototypeEner);
    double tWalltime = MPI_Wtime() - tStart; 

    Tile_cpu_tf_dens::releaseScratch();
    Tile_cpu_tf_ener::releaseScratch();

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
    computeLaplacianFused_cpu.routine         = cpu_tf_fused::taskFunction;

    Tile_cpu_tf_fused::acquireScratch();
    const Tile_cpu_tf_fused    prototype{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuTasks("Fused Kernels CPU",
                                        computeLaplacianFused_cpu, prototype);
    double tWalltime = MPI_Wtime() - tStart; 

    Tile_cpu_tf_fused::releaseScratch();

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Kernels - Host");
}

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

    const DataPacket_gpu_tf_dens&   packetPrototypeDens{};
    const DataPacket_gpu_tf_ener&   packetPrototypeEner{}; 
    
    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("LapDens", 1, 0, computeLaplacianDensity,
                                        packetPrototypeDens);
    Runtime::instance().executeGpuTasks("LapEner", 1, 0, computeLaplacianEnergy,
                                        packetPrototypeEner);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End TestGpuOnlyConfig");
}

TEST_F(TestRuntime, TestCpuGpuConfig) {
    Logger::instance().log("[googletest] Start TestCpuGpu");

    RuntimeAction    computeLaplacianDensity;
    RuntimeAction    computeLaplacianEnergy;

    computeLaplacianDensity.name            = "LaplacianDensity";
    computeLaplacianDensity.nInitialThreads = 3;
    computeLaplacianDensity.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianDensity.nTilesPerPacket = 0;
    computeLaplacianDensity.routine         = cpu_tf_dens::taskFunction;

    computeLaplacianEnergy.name            = "LaplacianEnergy";
    computeLaplacianEnergy.nInitialThreads = 3;
    computeLaplacianEnergy.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianEnergy.nTilesPerPacket = 20;
    computeLaplacianEnergy.routine         = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    Tile_cpu_tf_dens::acquireScratch();

    const Tile_cpu_tf_dens    tilePrototypeDens{};
    const DataPacket_gpu_tf_ener&   packetPrototypeEner{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuTasks("ConcurrentCpuGpu",
                                           computeLaplacianDensity,
                                           tilePrototypeDens,
                                           computeLaplacianEnergy,
                                           packetPrototypeEner);
    double tWalltime = MPI_Wtime() - tStart; 

    Tile_cpu_tf_dens::releaseScratch();

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End TestCpuGpu");
}

// TODO: This test uses 2 packets, dens and ener packets.  
TEST_F(TestRuntime, TestSharedCpuGpuConfig) {
    constexpr unsigned int   N_DIST_THREADS = 2;

    Logger::instance().log("[googletest] Start Data Parallel Cpu/Gpu");

    RuntimeAction    computeLaplacian_cpu;
    RuntimeAction    computeLaplacian_gpu;

    computeLaplacian_cpu.name            = "LaplacianDensity_cpu";
    computeLaplacian_cpu.nInitialThreads = 4;
    computeLaplacian_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacian_cpu.nTilesPerPacket = 0;
    computeLaplacian_cpu.routine         = cpu_tf_dens::taskFunction;

    computeLaplacian_gpu.name            = "LaplacianDensity_gpu";
    computeLaplacian_gpu.nInitialThreads = 2;
    computeLaplacian_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacian_gpu.nTilesPerPacket = 30;
    computeLaplacian_gpu.routine         = ActionRoutines::computeLaplacianDensity_packet_oacc_summit;

    Tile_cpu_tf_dens::acquireScratch();
    Tile_cpu_tf_ener::acquireScratch();

    const Tile_cpu_tf_dens           tilePrototypeDens{};
    const Tile_cpu_tf_ener           tilePrototypeEner{};
    const DataPacket_gpu_tf_dens&   packetPrototypeDens{};
    const DataPacket_gpu_tf_ener&   packetPrototypeEner{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuSplitTasks("DataParallelDensity",
                                                N_DIST_THREADS, 0,
                                                computeLaplacian_cpu,
                                                tilePrototypeDens,
                                                computeLaplacian_gpu,
                                                packetPrototypeDens,
                                                30);

    computeLaplacian_cpu.name    = "LaplacianEnergy_cpu";
    computeLaplacian_cpu.routine = cpu_tf_ener::taskFunction;

    computeLaplacian_gpu.name    = "LaplacianEnergy_gpu";
    computeLaplacian_gpu.routine = ActionRoutines::computeLaplacianEnergy_packet_oacc_summit;

    Runtime::instance().executeCpuGpuSplitTasks("DataParallelEnergy",
                                                N_DIST_THREADS, 0,
                                                computeLaplacian_cpu,
                                                tilePrototypeEner,
                                                computeLaplacian_gpu,
                                                packetPrototypeEner,
                                                30);
    double tWalltime = MPI_Wtime() - tStart; 

    Tile_cpu_tf_dens::releaseScratch();
    Tile_cpu_tf_ener::releaseScratch();

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Data Parallel Cpu/Gpu");
}

TEST_F(TestRuntime, TestSharedCpuGpuWowza) {
    Logger::instance().log("[googletest] Start Cpu/Gpu Wowza Config");

    RuntimeAction    computeLaplacianDensity_cpu;
    RuntimeAction    computeLaplacianDensity_gpu;
    RuntimeAction    computeLaplacianEnergy_gpu;

    computeLaplacianDensity_cpu.name            = "LaplacianDensity_cpu";
    computeLaplacianDensity_cpu.nInitialThreads = 2;
    computeLaplacianDensity_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacianDensity_cpu.nTilesPerPacket = 0;
    computeLaplacianDensity_cpu.routine         = cpu_tf_dens::taskFunction;

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

    Tile_cpu_tf_dens::acquireScratch();

    const Tile_cpu_tf_dens           tilePrototypeDens{};
    const DataPacket_gpu_tf_dens& packetPrototypeDens{};
    const DataPacket_gpu_tf_ener& packetPrototypeEner{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuWowzaTasks("CPU/GPU Wowza",
                                                computeLaplacianDensity_cpu,
                                                tilePrototypeDens,
                                                computeLaplacianDensity_gpu,
                                                computeLaplacianEnergy_gpu,
                                                packetPrototypeDens,
                                                packetPrototypeEner,
                                                20);
    double tWalltime = MPI_Wtime() - tStart; 

    Tile_cpu_tf_dens::releaseScratch();

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Cpu/Gpu Wowza Config");
}

TEST_F(TestRuntime, TestFusedActions) {
    Logger::instance().log("[googletest] Start Fused Actions");

    RuntimeAction    computeLaplacianFused_gpu;

    computeLaplacianFused_gpu.name            = "LaplacianFusedActions_gpu";
    computeLaplacianFused_gpu.nInitialThreads = 3;
    computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianFused_gpu.nTilesPerPacket = 20;
    computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit;

    const DataPacket_gpu_tf_fused_actions&   packetPrototype{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("Fused Actions GPU", 1, 0, computeLaplacianFused_gpu,
                                        packetPrototype);
    double tWalltime = MPI_Wtime() - tStart; 

//    std::vector<std::string>    names{};
//    names.push_back("density");
//    names.push_back("energy");
//    Grid::instance().writePlotfile("Wowza", names);

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Actions");
}

TEST_F(TestRuntime, TestFusedKernelsStrong) {
    Logger::instance().log("[googletest] Start Fused Kernels Strong");

    RuntimeAction    computeLaplacianFused_gpu;

    computeLaplacianFused_gpu.name            = "LaplacianFusedKernelsStrong_gpu";
    computeLaplacianFused_gpu.nInitialThreads = 3;
    computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianFused_gpu.nTilesPerPacket = 20;
    computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsStrong_packet_oacc_summit;

    const DataPacket_gpu_tf_fused_kernels&   packetPrototype{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("Fused Kernels Strong GPU", 1, 0, computeLaplacianFused_gpu,
                                        packetPrototype);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Kernels Strong");
}

TEST_F(TestRuntime, TestFusedKernelsWeak) {
    Logger::instance().log("[googletest] Start Fused Kernels Weak");

    RuntimeAction    computeLaplacianFused_gpu;

    computeLaplacianFused_gpu.name            = "LaplacianFusedKernelsWeak_gpu";
    computeLaplacianFused_gpu.nInitialThreads = 3;
    computeLaplacianFused_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacianFused_gpu.nTilesPerPacket = 20;
    computeLaplacianFused_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsWeak_packet_oacc_summit;

    const DataPacket_gpu_tf_fused_kernels&   packetPrototype{}; 

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeGpuTasks("Fused Kernels Weak GPU", 1, 0, computeLaplacianFused_gpu,
                                        packetPrototype);
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Kernels Weak");
}

TEST_F(TestRuntime, TestSharedCpuGpuConfigFusedActions) {
    constexpr unsigned int   N_DIST_THREADS = 2;

    Logger::instance().log("[googletest] Start Data Parallel Cpu/Gpu Fused Actions");

    RuntimeAction    computeLaplacian_cpu;
    RuntimeAction    computeLaplacian_gpu;

    computeLaplacian_cpu.name            = "LaplacianFused_cpu";
    computeLaplacian_cpu.nInitialThreads = 3;
    computeLaplacian_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacian_cpu.nTilesPerPacket = 0;
    computeLaplacian_cpu.routine         = cpu_tf_fused::taskFunction;

    computeLaplacian_gpu.name            = "LaplacianFused_gpu";
    computeLaplacian_gpu.nInitialThreads = 3;
    computeLaplacian_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacian_gpu.nTilesPerPacket = 20;
    computeLaplacian_gpu.routine         = ActionRoutines::computeLaplacianFusedActions_packet_oacc_summit;

    Tile_cpu_tf_fused::acquireScratch();

    const Tile_cpu_tf_fused   tilePrototype{};
    const DataPacket_gpu_tf_fused_actions&   packetPrototype{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuSplitTasks("DataParallelFused",
                                                N_DIST_THREADS, 0,
                                                computeLaplacian_cpu,
                                                tilePrototype,
                                                computeLaplacian_gpu,
                                                packetPrototype,
                                                15);
    double tWalltime = MPI_Wtime() - tStart; 

    Tile_cpu_tf_fused::releaseScratch();

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Data Parallel Cpu/Gpu Fused Actions");
}

TEST_F(TestRuntime, TestSharedCpuGpuConfigFusedKernels) {
    constexpr unsigned int   N_DIST_THREADS = 2;

    Logger::instance().log("[googletest] Start Data Parallel Cpu/Gpu Fused Kernels");

    RuntimeAction    computeLaplacian_cpu;
    RuntimeAction    computeLaplacian_gpu;

    computeLaplacian_cpu.name            = "LaplacianFused_cpu";
    computeLaplacian_cpu.nInitialThreads = 4;
    computeLaplacian_cpu.teamType        = ThreadTeamDataType::BLOCK;
    computeLaplacian_cpu.nTilesPerPacket = 0;
    computeLaplacian_cpu.routine         = cpu_tf_fused::taskFunction;

    computeLaplacian_gpu.name            = "LaplacianFused_gpu";
    computeLaplacian_gpu.nInitialThreads = 2;
    computeLaplacian_gpu.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    computeLaplacian_gpu.nTilesPerPacket = 20;
    computeLaplacian_gpu.routine         = ActionRoutines::computeLaplacianFusedKernelsStrong_packet_oacc_summit;

    Tile_cpu_tf_fused::acquireScratch();

    const Tile_cpu_tf_fused   tilePrototype{};
    const DataPacket_gpu_tf_fused_kernels&   packetPrototype{};

    double tStart = MPI_Wtime(); 
    Runtime::instance().executeCpuGpuSplitTasks("DataParallelFused",
                                                N_DIST_THREADS, 0,
                                                computeLaplacian_cpu,
                                                tilePrototype,
                                                computeLaplacian_gpu,
                                                packetPrototype,
                                                15);
    double tWalltime = MPI_Wtime() - tStart; 

    Tile_cpu_tf_fused::releaseScratch();

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Data Parallel Cpu/Gpu Fused Kernels");
}

}

