#include <stdio.h>
#include <cmath>
#include <cassert>
#include <array>
#include <vector>
#include <pthread.h>

#include "constants.h"
#include "Block.h"
#include "Grid.h"
#include "BlockIterator.h"
#include "OrchestrationRuntime.h"

#include "scaleEnergy_cpu.h"
#include "computeLaplacianDensity_cpu.h"
#include "computeLaplacianEnergy_cpu.h"

#include "gtest/gtest.h"

namespace {

/**
 * Define a test fixture
 */
class TestRuntimeBlock : public testing::Test {
protected:
    // TASK_COMPOSER: The offline tool will need to determine how many thread
    // teams are needed as well as how many threads to allocate to each.
    static constexpr unsigned int N_THREAD_TEAMS = 3;
    static constexpr unsigned int MAX_THREADS    = 5;

    static constexpr double       X_MIN      = 0.0;
    static constexpr double       X_MAX      = 1.0;
    static constexpr double       Y_MIN      = 0.0;
    static constexpr double       Y_MAX      = 1.0;
    static constexpr unsigned int N_GUARD    = 1;
    static constexpr unsigned int NXB        = 8;
    static constexpr unsigned int NYB        = 16;
    static constexpr unsigned int MAX_BLOCKS = 100;

    OrchestrationRuntime<Block>*   runtime_;

    TestRuntimeBlock(void) {
        OrchestrationRuntime<Block>::setNumberThreadTeams(N_THREAD_TEAMS);
        OrchestrationRuntime<Block>::setMaxThreadsPerTeam(MAX_THREADS);
        runtime_ = OrchestrationRuntime<Block>::instance();
    }

    ~TestRuntimeBlock(void) {
        delete OrchestrationRuntime<Block>::instance();
    }

    /**
     * PROBLEM ONE
     *      Approximated exactly by second-order discretized Laplacian
     */
    static double f1(const double x, const double y) {
        return (  3.0*x*x*x +     x*x + x 
                - 2.0*y*y*y - 1.5*y*y + y
                + 5.0);
    }

    static double Delta_f1(const double x, const double y) {
        return (18.0*x - 12.0*y - 1.0);
    }

    /**
     * PROBLEM TWO
     *      Approximation is not exact and we know the error term exactly */
    static double f2(const double x, const double y) {
        return (  4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                + 1.0);
    }

    static double Delta_f2(const double x, const double y) {
        return (  48.0*x*x - 18.0*x
                - 12.0*y*y + 12.0*y
                - 2.0); 
    }

    void initializeData(BlockIterator& itor) {
        int       i0   = 0;
        int       j0   = 0;
        double    x    = 0.0;
        double*** data = NULL;
    
        std::vector<double>   x_coords;
        std::vector<double>   y_coords;
        std::array<int,NDIM>  loGC;
        std::array<int,NDIM>  hiGC;
    
        Block block;
        for (itor.clear(); itor.isValid(); itor.next()) {
             block = itor.currentBlock();
     
             data     = block.dataPtr();
             loGC     = block.loGC();
             hiGC     = block.hiGC();
             x_coords = block.coordinates(IAXIS);
             y_coords = block.coordinates(JAXIS);
    
             i0 = loGC[IAXIS];
             j0 = loGC[JAXIS];
             for      (int i=loGC[IAXIS]; i<=hiGC[IAXIS]; ++i) {
                  x = x_coords[i-i0];
                  for (int j=loGC[JAXIS]; j<=hiGC[JAXIS]; ++j) { 
                       data[DENS_VAR][i-i0][j-j0] = f1(x, y_coords[j-j0]);
                       data[ENER_VAR][i-i0][j-j0] = f2(x, y_coords[j-j0]);
                  }
             }
        }
    }

    void computeError(BlockIterator& itor, double* L_inf1, double* meanAbsErr1,
                                           double* L_inf2, double* meanAbsErr2) {
        double***                     data = NULL;
        std::vector<double>           x_coords;
        std::vector<double>           y_coords;
        std::array<unsigned int,NDIM> lo;
        std::array<unsigned int,NDIM> hi;
        std::array<int,NDIM>          loGC;
    
        int i0              = 0;
        int j0              = 0;
        double x            = 0.0;
        double y            = 0.0;
        double absErr       = 0.0;
        double maxAbsErr1   = 0.0;
        double sum1         = 0.0;
        double maxAbsErr2   = 0.0;
        double sum2         = 0.0;
        unsigned int nCells = 0;
    
        Block block;
        for (itor.clear(); itor.isValid(); itor.next()) {
             block = itor.currentBlock();
     
             data     = block.dataPtr();
             lo       = block.lo();
             hi       = block.hi();
             loGC     = block.loGC();
             x_coords = block.coordinates(IAXIS);
             y_coords = block.coordinates(JAXIS);
    
             i0 = loGC[IAXIS];
             j0 = loGC[JAXIS];
             for      (int i=lo[IAXIS]; i<=hi[IAXIS]; ++i) {
                  x = x_coords[i-i0];
                  for (int j=lo[JAXIS]; j<=hi[JAXIS]; ++j) { 
                       y = y_coords[j-j0];
    
                       absErr = fabs(Delta_f1(x, y) - data[DENS_VAR][i-i0][j-j0]);
                       sum1 += absErr;
                       if (absErr > maxAbsErr1) {
                            maxAbsErr1 = absErr;
                       }
    
                       absErr = fabs(3.2*Delta_f2(x, y) - data[ENER_VAR][i-i0][j-j0]);
                       sum2 += absErr;
                       if (absErr > maxAbsErr2) {
                            maxAbsErr2 = absErr;
                       }
    
                       ++nCells;
                  }
             }
        }
    
        *L_inf1 = maxAbsErr1;
        *meanAbsErr1 = sum1 / ((double)nCells);
    
        *L_inf2 = maxAbsErr2;
        *meanAbsErr2 = sum2 / ((double)nCells);
    }
};

void cpuNoop(const unsigned int tId,
             const std::string& name,
             Block& work) { }

#ifndef VERBOSE
TEST_F(TestRuntimeBlock, TestSingleTeam) {
    constexpr unsigned int  N_THREADS      = 4;
    constexpr unsigned int  N_BLOCKS_X     = 512;
    constexpr unsigned int  N_BLOCKS_Y     = 256;

    ThreadTeam<Block>   cpu(4, 1, "TestSingleTeam.log");

    //***** SETUP DOMAIN MESH
    Grid myGrid(X_MIN, X_MAX, Y_MIN, Y_MAX,
                NXB, NYB, N_BLOCKS_X, N_BLOCKS_Y,
                N_GUARD, N_VARIABLES);

    // The test problem assumes square mesh
    std::array<double,NDIM> deltas = myGrid.deltas();
    ASSERT_EQ(deltas[0], deltas[1]);
        
    BlockIterator itor(&myGrid);
    initializeData(itor);

    try {
        cpu.startTask(ThreadRoutines::computeLaplacianEnergy_cpu, N_THREADS,
                      "Cpu", "LaplacianEnergy");
        for (itor.clear(); itor.isValid(); itor.next()) {
             cpu.enqueue(itor.currentBlock());
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::computeLaplacianDensity_cpu, N_THREADS,
                      "Cpu", "LaplacianDensity");
        for (itor.clear(); itor.isValid(); itor.next()) {
             cpu.enqueue(itor.currentBlock());
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::scaleEnergy_cpu, N_THREADS,
                      "Cpu", "scaleEnergy");
        for (itor.clear(); itor.isValid(); itor.next()) {
             cpu.enqueue(itor.currentBlock());
        }
        cpu.closeTask();
        cpu.wait();
    } catch (std::invalid_argument  e) {
        printf("\nINVALID ARGUMENT: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (std::logic_error  e) {
        printf("\nLOGIC ERROR: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (std::runtime_error  e) {
        printf("\nRUNTIME ERROR: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (...) {
        printf("\n??? ERROR: Unanticipated error\n\n");
        EXPECT_TRUE(false);
    }

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    computeError(itor, &L_inf1, &meanAbsErr1, &L_inf2, &meanAbsErr2);

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

#ifndef VERBOSE
TEST_F(TestRuntimeBlock, TestRuntimeSingle) {
    constexpr unsigned int  N_BLOCKS_X     = 512;
    constexpr unsigned int  N_BLOCKS_Y     = 256;

    ThreadTeam<Block>   cpu(4, 1, "TestRuntimeSingle.log");

    //***** SETUP DOMAIN MESH
    Grid myGrid(X_MIN, X_MAX, Y_MIN, Y_MAX,
                NXB, NYB, N_BLOCKS_X, N_BLOCKS_Y,
                N_GUARD, N_VARIABLES);

    // The test problem assumes square mesh
    std::array<double,NDIM> deltas = myGrid.deltas();
    ASSERT_EQ(deltas[0], deltas[1]);
        
    BlockIterator itor(&myGrid);
    initializeData(itor);

    try {
        // Give an extra thread to the GPU task so that it can start to get work
        // to the postGpu task quicker.
        runtime_->executeTask(myGrid, "Task Bundle 1",
                              ThreadRoutines::computeLaplacianDensity_cpu,
                              1, "bundle1_cpuTask",
                              ThreadRoutines::computeLaplacianEnergy_cpu,
                              2, "bundle1_gpuTask",
                              ThreadRoutines::scaleEnergy_cpu,
                              0, "bundle1_postGpuTask");
    } catch (std::invalid_argument  e) {
        printf("\nINVALID ARGUMENT: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (std::logic_error  e) {
        printf("\nLOGIC ERROR: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (std::runtime_error  e) {
        printf("\nRUNTIME ERROR: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (...) {
        printf("\n??? ERROR: Unanticipated error\n\n");
        EXPECT_TRUE(false);
    }

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    computeError(itor, &L_inf1, &meanAbsErr1, &L_inf2, &meanAbsErr2);

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

#ifndef VERBOSE
TEST_F(TestRuntimeBlock, TestRuntimeScaling) {
    const unsigned int N_TRIALS = 7;
    const unsigned int N_BLOCKS_X_ALL[N_TRIALS] = {8, 16, 32, 64, 128, 256, 512};
    const unsigned int N_BLOCKS_Y_ALL[N_TRIALS] = {4,  8, 16, 32,  64, 128, 256};

    std::array<double,NDIM>   deltas;

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    double results[N_TRIALS][3];
    for (int i=0; i<N_TRIALS; ++i) {
        //***** SETUP DOMAIN MESH
        Grid myGrid(X_MIN, X_MAX, Y_MIN, Y_MAX,
                    NXB, NYB, N_BLOCKS_X_ALL[i], N_BLOCKS_Y_ALL[i],
                    N_GUARD, N_VARIABLES);

        // The test problem assumes square mesh
        deltas      = myGrid.deltas();
        ASSERT_EQ(deltas[0], deltas[1]);

        BlockIterator itor(&myGrid);
        initializeData(itor);

        //***** COMPUTATION STAGE
        // At this level, the Orchestration System has been used to contruct the
        // operation function that we call.  Hopefully, the interface of this
        // function will not vary with the Orchestration Runtime chosen at setup.
        // So far, the Driver is Runtime agnostic.
        try {
            // myGrid is a standin for the set of parameters we need to
            // specify the tile iterator to use.
            runtime_->executeTask(myGrid, "Task Bundle 1",
                                  ThreadRoutines::computeLaplacianDensity_cpu,
                                  1, "bundle1_cpuTask",
                                  ThreadRoutines::computeLaplacianEnergy_cpu,
                                  2, "bundle1_gpuTask",
                                  ThreadRoutines::scaleEnergy_cpu,
                                  0, "bundle1_postGpuTask");
        } catch (std::invalid_argument  e) {
            printf("\nINVALID ARGUMENT: %s\n\n", e.what());
            ASSERT_TRUE(false);
        } catch (std::logic_error  e) {
            printf("\nLOGIC ERROR: %s\n\n", e.what());
            ASSERT_TRUE(false);
        } catch (std::runtime_error  e) {
            printf("\nRUNTIME ERROR: %s\n\n", e.what());
            ASSERT_TRUE(false);
        } catch (...) {
            printf("\n??? ERROR: Unanticipated error\n\n");
            ASSERT_TRUE(false);
        }

        computeError(itor, &L_inf1, &meanAbsErr1, &L_inf2, &meanAbsErr2);

        // No scaling relationship for the Density data
        //    => Check now and don't store data
        EXPECT_TRUE(0.0 <= L_inf1);
        EXPECT_TRUE(L_inf1 <= 1.0e-15);
        EXPECT_TRUE(0.0 <= meanAbsErr1);
        EXPECT_TRUE(meanAbsErr1 <= 1.0e-15);

        results[i][0] = NXB * N_BLOCKS_X_ALL[i];
        results[i][1] = L_inf2;
        results[i][2] = meanAbsErr2;
    }

    // Check energy scaling
    double       deltaErr   = 0.0;
    double       deltaCells = 0.0;
    for (int i=1; i<N_TRIALS; ++i) {
        deltaErr   = log10(results[i][2]) -log10(results[i-1][2]);
        deltaCells = log10(results[i][0]) -log10(results[i-1][0]);

        // Confirm that we get second-order scaling
        EXPECT_NEAR(-2.0, deltaErr / deltaCells, 1.0e-5);
    }

    // Confirm that we also get reasonable absolute error
    EXPECT_TRUE(0.0 <= results[N_TRIALS-1][1]);
    EXPECT_TRUE(results[N_TRIALS-1][1] <= 5.0e-6);
    EXPECT_TRUE(0.0 <= results[N_TRIALS-1][2]);
    EXPECT_TRUE(results[N_TRIALS-1][2] <= 5.0e-6);
}
#endif

}
