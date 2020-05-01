#include <cmath>
#include <cassert>
#include <array>
#include <vector>
#include <iostream>
#include <pthread.h>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Array4.H>

#include "Tile.h"
#include "Grid.h"
#include "ThreadTeam.h"
#include "OrchestrationRuntime.h"

#include "Flash.h"
#include "constants.h"
#include "Analysis.h"
#include "setInitialConditions_block.h"
#include "scaleEnergy_block.h"
#include "computeLaplacianDensity_block.h"
#include "computeLaplacianEnergy_block.h"

#include "gtest/gtest.h"

namespace {

/**
 * Define a test fixture
 */
class TestRuntimeTile : public testing::Test {
protected:
    // TASK_COMPOSER: The offline tool will need to determine how many thread
    // teams are needed as well as how many threads to allocate to each.
    static constexpr unsigned int   N_TILE_THREAD_TEAMS   = 3;
    static constexpr unsigned int   N_PACKET_THREAD_TEAMS = 0;
    static constexpr unsigned int   MAX_THREADS    = 5;

    OrchestrationRuntime*   runtime_;

    TestRuntimeTile(void) {
        OrchestrationRuntime::setNumberThreadTeams(N_TILE_THREAD_TEAMS,
                                                   N_PACKET_THREAD_TEAMS);
        OrchestrationRuntime::setMaxThreadsPerTeam(MAX_THREADS);
        runtime_ = OrchestrationRuntime::instance();

        Grid*    grid = Grid::instance();
        grid->initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                         N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                         NUNKVAR,
                         Simulation::setInitialConditions_block);
   }

    ~TestRuntimeTile(void) {
        delete OrchestrationRuntime::instance();
        Grid::instance()->destroyDomain();
    }
};

#ifndef DEBUG_RUNTIME
TEST_F(TestRuntimeTile, TestSingleTeam) {
    amrex::MultiFab&   unk = Grid::instance()->unk();

    constexpr unsigned int  N_THREADS = 4;
    ThreadTeam<Tile>  cpu(N_THREADS, 1, "TestSingleTeam.log");

    // Fix simulation to a single level and use AMReX 0-based indexing
    unsigned int   level = 0;

    try {
        cpu.startTask(ThreadRoutines::computeLaplacianEnergy_block, N_THREADS,
                      "Cpu", "LaplacianEnergy");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile   myTile(itor, level);
            cpu.enqueue(myTile, true);
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::computeLaplacianDensity_block, N_THREADS,
                      "Cpu", "LaplacianDensity");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile   myTile(itor, level);
            cpu.enqueue(myTile, true);
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::scaleEnergy_block, N_THREADS,
                      "Cpu", "scaleEnergy");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile   myTile(itor, level);
            cpu.enqueue(myTile, true);
        }
        cpu.closeTask();
        cpu.wait();

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        cpu.startTask(Analysis::computeErrors_block, N_THREADS,
                      "Analysis", "computeErrors");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile   myTile(itor, level);
            cpu.enqueue(myTile, true);
        }
        cpu.closeTask();
        cpu.wait();
    } catch (std::invalid_argument  e) {
        std::cerr << "\nINVALID ARGUMENT: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::logic_error  e) {
        std::cerr << "\nLOGIC ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::runtime_error  e) {
        std::cerr << "\nRUNTIME ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (...) {
        std::cerr << "\n??? ERROR: Unanticipated error\n\n";
        EXPECT_TRUE(false);
    }

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    Analysis::densityErrors(&L_inf1, &meanAbsErr1);
    Analysis::energyErrors(&L_inf2, &meanAbsErr2);

    EXPECT_TRUE(0.0 <= L_inf1);
    EXPECT_TRUE(L_inf1 <= 0.0);
    EXPECT_TRUE(0.0 <= meanAbsErr1);
    EXPECT_TRUE(meanAbsErr1 <= 0.0);

    EXPECT_TRUE(0.0 <= L_inf2);
    EXPECT_TRUE(L_inf2 <= 9.0e-6);
    EXPECT_TRUE(0.0 <= meanAbsErr2);
    EXPECT_TRUE(meanAbsErr2 <= 9.0e-6);
}
#endif

#ifndef DEBUG_RUNTIME
TEST_F(TestRuntimeTile, TestRuntimeSingle) {
    ActionBundle    bundle;

    try {
        // Give an extra thread to the GPU task so that it can start to get work
        // to the postGpu task quicker.
        bundle.name                          = "Action Bundle 1";
        bundle.cpuAction.name                = "bundle1_cpuAction";
        bundle.cpuAction.nInitialThreads     = 1;
        bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.cpuAction.routine             = ThreadRoutines::computeLaplacianDensity_block;
        bundle.gpuAction.name                = "bundle1_gpuAction";
        bundle.gpuAction.nInitialThreads     = 2;
        bundle.gpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.gpuAction.routine             = ThreadRoutines::computeLaplacianEnergy_block;
        bundle.postGpuAction.name            = "bundle1_postGpuAction";
        bundle.postGpuAction.nInitialThreads = 0;
        bundle.postGpuAction.teamType        = ThreadTeamDataType::BLOCK;
        bundle.postGpuAction.routine         = ThreadRoutines::scaleEnergy_block;

        runtime_->executeTasks(bundle);

        bundle.name                          = "Analysis Bundle";
        bundle.cpuAction.name                = "computeErrors";
        bundle.cpuAction.nInitialThreads     = 2;
        bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.cpuAction.routine             = Analysis::computeErrors_block;
        bundle.gpuAction.name                = "";
        bundle.gpuAction.nInitialThreads     = 0;
        bundle.gpuAction.routine             = nullptr;
        bundle.postGpuAction.name            = "";
        bundle.postGpuAction.nInitialThreads = 0;
        bundle.postGpuAction.routine         = nullptr;

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        runtime_->executeTasks(bundle);
    } catch (std::invalid_argument  e) {
        std::cerr << "\nINVALID ARGUMENT: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::logic_error  e) {
        std::cerr << "\nLOGIC ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::runtime_error  e) {
        std::cerr << "\nRUNTIME ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (...) {
        std::cerr << "\n??? ERROR: Unanticipated error\n\n";
        EXPECT_TRUE(false);
    }

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    Analysis::densityErrors(&L_inf1, &meanAbsErr1);
    Analysis::energyErrors(&L_inf2, &meanAbsErr2);
//    std::cout << "L_inf1 = " << L_inf1 << "\n";
//    std::cout << "L_inf2 = " << L_inf2 << std::endl;

    EXPECT_TRUE(0.0 <= L_inf1);
    EXPECT_TRUE(L_inf1 <= 1.0e-15);
    EXPECT_TRUE(0.0 <= meanAbsErr1);
    EXPECT_TRUE(meanAbsErr1 <= 1.0e-15);

    EXPECT_TRUE(0.0 <= L_inf2);
    EXPECT_TRUE(L_inf2 <= 9.0e-6);
    EXPECT_TRUE(0.0 <= meanAbsErr2);
    EXPECT_TRUE(meanAbsErr2 <= 9.0e-6);
}
#endif

}

