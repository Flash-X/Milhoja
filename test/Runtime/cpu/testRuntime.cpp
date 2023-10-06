#include <gtest/gtest.h>

#include <Milhoja_Logger.h>
#include <Milhoja_Grid.h>
#include <Milhoja_axis.h>
#include <Milhoja_edge.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_TileWrapper.h>
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

using namespace milhoja;

namespace {

class TestRuntime : public testing::Test {
protected:
    TestRuntime(void) {
        // TODO: Set with Mihoja?
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
        Tile_cpu_tf_analysis   prototype{};
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

    double tStart = MPI_Wtime(); 
    Tile_cpu_tf_dens::acquireScratch();
    Tile_cpu_tf_dens    prototypeDens{};
    Runtime::instance().executeCpuTasks("LapDens",
                                        computeLaplacianDensity, prototypeDens);
    Tile_cpu_tf_dens::releaseScratch();

    Tile_cpu_tf_ener::acquireScratch();
    Tile_cpu_tf_ener    prototypeEner{};
    Runtime::instance().executeCpuTasks("LapEner",
                                        computeLaplacianEnergy, prototypeEner);
    Tile_cpu_tf_ener::releaseScratch();
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
    computeLaplacianFused_cpu.routine         = cpu_tf_fused::taskFunction;

    double tStart = MPI_Wtime(); 
    Tile_cpu_tf_fused::acquireScratch();
    Tile_cpu_tf_fused    prototype{};
    Runtime::instance().executeCpuTasks("Fused Kernels CPU",
                                        computeLaplacianFused_cpu, prototype);
    Tile_cpu_tf_fused::releaseScratch();
    double tWalltime = MPI_Wtime() - tStart; 

    checkSolution();
    std::cout << "Total walltime = " << tWalltime << " sec\n";

    Logger::instance().log("[googletest] End Fused Kernels - Host");
}

}

