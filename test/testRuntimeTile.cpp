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
#include "scaleEnergy_cpu.h"
#include "computeLaplacianDensity_cpu.h"
#include "computeLaplacianEnergy_cpu.h"

#include "gtest/gtest.h"

namespace {

/**
 * Define a test fixture
 */
class TestRuntimeTile : public testing::Test {
protected:
    // TASK_COMPOSER: The offline tool will need to determine how many thread
    // teams are needed as well as how many threads to allocate to each.
    static constexpr unsigned int   N_THREAD_TEAMS = 3;
    static constexpr unsigned int   MAX_THREADS    = 5;

    OrchestrationRuntime*   runtime_;

    TestRuntimeTile(void) {
        OrchestrationRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
        OrchestrationRuntime::setMaxThreadsPerTeam(MAX_THREADS);
        runtime_ = OrchestrationRuntime::instance();

        Grid<NXB,NYB,NZB,NGUARD>*    grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
        grid->initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                         N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                         NUNKVAR,
                         initBlock);
   }

    ~TestRuntimeTile(void) {
        delete OrchestrationRuntime::instance();
        Grid<NXB,NYB,NZB,NGUARD>::instance()->destroyDomain();
    }

    static void initBlock(Tile* tileDesc) {
        Grid<NXB,NYB,NZB,NGUARD>*    grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
        amrex::Geometry     geometry = grid->geometry();
        amrex::MultiFab&    unk = grid->unk();
        amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];
        grid = nullptr;

        // TODO: Make getting data ptr in C++ a method in Tile?
        amrex::Array4<amrex::Real> const&   f = fab.array();

        // Fill in the GC data as well as we aren't doing a GC fill in any
        // of these tests
        amrex::Real   x = 0.0;
        amrex::Real   y = 0.0;
        const amrex::Dim3 loGC = tileDesc->loGC();
        const amrex::Dim3 hiGC = tileDesc->hiGC();
        for     (int j = loGC.y; j <= hiGC.y; ++j) {
            y = geometry.CellCenter(j, 1);
            for (int i = loGC.x; i <= hiGC.x; ++i) {
                x = geometry.CellCenter(i, 0);
                // PROBLEM ONE
                //  Approximated exactly by second-order discretized Laplacian
                f(i, j, loGC.z, DENS_VAR_C) =   3.0*x*x*x +     x*x + x 
                                              - 2.0*y*y*y - 1.5*y*y + y
                                              + 5.0;
                // PROBLEM TWO
                //  Approximation is not exact and we know the error term exactly
                f(i, j, loGC.z, ENER_VAR_C) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                                              -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                              + 1.0;
            }
        }
    }
};

#ifndef DEBUG_RUNTIME
TEST_F(TestRuntimeTile, TestSingleTeam) {
    amrex::MultiFab&   unk = Grid<NXB,NYB,NZB,NGUARD>::instance()->unk();

    constexpr unsigned int  N_THREADS = 4;
    ThreadTeam<Tile>  cpu(N_THREADS, 1, "TestSingleTeam.log");

    // Fix simulation to a single level and use AMReX 0-based indexing
    unsigned int   level = 0;

    try {
        cpu.startTask(ThreadRoutines::computeLaplacianEnergy_cpu, N_THREADS,
                      "Cpu", "LaplacianEnergy");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile   myTile(itor, level);
            cpu.enqueue(myTile, true);
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::computeLaplacianDensity_cpu, N_THREADS,
                      "Cpu", "LaplacianDensity");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile   myTile(itor, level);
            cpu.enqueue(myTile, true);
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::scaleEnergy_cpu, N_THREADS,
                      "Cpu", "scaleEnergy");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile   myTile(itor, level);
            cpu.enqueue(myTile, true);
        }
        cpu.closeTask();
        cpu.wait();

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        cpu.startTask(Analysis::computeErrors, N_THREADS,
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
    EXPECT_TRUE(L_inf2 <= 5.0e-6);
    EXPECT_TRUE(0.0 <= meanAbsErr2);
    EXPECT_TRUE(meanAbsErr2 <= 5.0e-6);
}
#endif

#ifndef DEBUG_RUNTIME
TEST_F(TestRuntimeTile, TestRuntimeSingle) {
    try {
        // Give an extra thread to the GPU task so that it can start to get work
        // to the postGpu task quicker.
        runtime_->executeTasks("Task Bundle 1",
                               ThreadRoutines::computeLaplacianDensity_cpu,
                               1, "bundle1_cpuTask",
                               ThreadRoutines::computeLaplacianEnergy_cpu,
                               2, "bundle1_gpuTask",
                               ThreadRoutines::scaleEnergy_cpu,
                               0, "bundle1_postGpuTask");

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        runtime_->executeTasks("Analysis bundle",
                               Analysis::computeErrors, 2, "bundle1_cpuTask",
                               nullptr, 0, "null_gpuTask",
                               nullptr, 0, "null_postGpuTask");
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

    // Output results to file for offline, manual scaling test
    std::string  fname("RuntimeCppTest_");
    fname += std::to_string(N_BLOCKS_X);
    fname += "_";
    fname += std::to_string(N_BLOCKS_Y);
    fname += ".dat";
    Analysis::writeToFile(fname);

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    Analysis::densityErrors(&L_inf1, &meanAbsErr1);
    Analysis::energyErrors(&L_inf2, &meanAbsErr2);

    EXPECT_TRUE(0.0 <= L_inf1);
    EXPECT_TRUE(L_inf1 <= 1.0e-15);
    EXPECT_TRUE(0.0 <= meanAbsErr1);
    EXPECT_TRUE(meanAbsErr1 <= 1.0e-15);

    EXPECT_TRUE(0.0 <= L_inf2);
    EXPECT_TRUE(L_inf2 <= 5.0e-6);
    EXPECT_TRUE(0.0 <= meanAbsErr2);
    EXPECT_TRUE(meanAbsErr2 <= 5.0e-6);
}
#endif

}

