#include <stdio.h>
#include <cmath>
#include <cassert>
#include <array>
#include <vector>
#include <pthread.h>

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_RealBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Array4.H>

#include "Tile.h"
#include "ThreadTeam.h"
#include "OrchestrationRuntime.h"

#include "constants.h"
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
    static constexpr unsigned int N_THREAD_TEAMS = 3;
    static constexpr unsigned int MAX_THREADS    = 5;

    static constexpr double       X_MIN      = 0.0;
    static constexpr double       X_MAX      = 1.0;
    static constexpr double       Y_MIN      = 0.0;
    static constexpr double       Y_MAX      = 1.0;
    static constexpr unsigned int N_GUARD    = 1;
    static constexpr unsigned int NXB        = 8;
    static constexpr unsigned int NYB        = 16;

    OrchestrationRuntime<Tile>*   runtime_;

    TestRuntimeTile(void) {
        OrchestrationRuntime<Tile>::setNumberThreadTeams(N_THREAD_TEAMS);
        OrchestrationRuntime<Tile>::setMaxThreadsPerTeam(MAX_THREADS);
        runtime_ = OrchestrationRuntime<Tile>::instance();
    }

    ~TestRuntimeTile(void) {
        delete OrchestrationRuntime<Tile>::instance();
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

    void initializeData(const amrex::Geometry& geometry, amrex::MultiFab&  mfab) {
        amrex::Real   x = 0.0;
        amrex::Real   y = 0.0;
        for (amrex::MFIter  itor(mfab); itor.isValid(); ++itor) {
            const amrex::Box&                    box = itor.fabbox();
            amrex::FArrayBox&                    fab = mfab[itor];
            amrex::Array4<amrex::Real> const&    data = fab.array();

            // Fill in the GC data as well as we aren't doing a GC fill in any
            // of these tests
            const amrex::Dim3 loGC = amrex::lbound(box);
            const amrex::Dim3 hiGC = amrex::ubound(box);
            for     (int j = loGC.y; j <= hiGC.y; ++j) {
                y = geometry.CellCenter(j, 1);
                for (int i = loGC.x; i <= hiGC.x; ++i) {
                    x = geometry.CellCenter(i, 0);
                    data(i, j, loGC.z, DENS_VAR) = f1(x, y);
                    data(i, j, loGC.z, ENER_VAR) = f2(x, y);
                }
            }
        }
    }

    void computeError(const amrex::Geometry& geometry, amrex::MultiFab& mfab,
                      amrex::Real* L_inf1, amrex::Real* meanAbsErr1,
                      amrex::Real* L_inf2, amrex::Real* meanAbsErr2) {
        amrex::Real  x            = 0.0;
        amrex::Real  y            = 0.0;
        amrex::Real  absErr       = 0.0;
        amrex::Real  maxAbsErr1   = 0.0;
        amrex::Real  sum1         = 0.0;
        amrex::Real  maxAbsErr2   = 0.0;
        amrex::Real  sum2         = 0.0;
        unsigned int nCells       = 0;

        for (amrex::MFIter  itor(mfab); itor.isValid(); ++itor) {
            const amrex::Box&                    box = itor.validbox();
            amrex::FArrayBox&                    fab = mfab[itor];
            amrex::Array4<amrex::Real> const&    data = fab.array();

            const amrex::Dim3 lo = amrex::lbound(box);
            const amrex::Dim3 hi = amrex::ubound(box);
            for     (int j = lo.y; j <= hi.y; ++j) {
                y = geometry.CellCenter(j, 1);
                for (int i = lo.x; i <= hi.x; ++i) {
                    x = geometry.CellCenter(i, 0);

                    absErr = fabs(Delta_f1(x, y) - data(i, j, lo.z, DENS_VAR));
                    sum1 += absErr;
                    if (absErr > maxAbsErr1) {
                         maxAbsErr1 = absErr;
                    }

                    absErr = fabs(3.2*Delta_f2(x, y) - data(i, j, lo.z, ENER_VAR));
                    sum2 += absErr;
                    if (absErr > maxAbsErr2) {
                         maxAbsErr2 = absErr;
                    }

                    ++nCells;
                }
            }
        }

        *L_inf1 = maxAbsErr1;
        *meanAbsErr1 = sum1 / static_cast<amrex::Real>(nCells);

        *L_inf2 = maxAbsErr2;
        *meanAbsErr2 = sum2 / static_cast<amrex::Real>(nCells);
    }
};

#ifndef VERBOSE
TEST_F(TestRuntimeTile, TestSingleTeam) {
    constexpr unsigned int  N_THREADS      = 4;
    constexpr unsigned int  N_BLOCKS_X     = 512;
    constexpr unsigned int  N_BLOCKS_Y     = 256;

    //***** SETUP DOMAIN, PROBLEM, and MESH
    amrex::IndexType    ccIndexSpace(amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
    amrex::IntVect      domainLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect      domainHi(AMREX_D_DECL(N_BLOCKS_X * NXB - 1,
                                              N_BLOCKS_Y * NYB - 1,
                                              0));
    amrex::Box          domain(domainLo, domainHi, ccIndexSpace);
    amrex::BoxArray     ba(domain);
    ba.maxSize(amrex::IntVect(AMREX_D_DECL(NXB, NYB, 1)));
    amrex::DistributionMapping  dm(ba);

    ASSERT_EQ(N_BLOCKS_X * N_BLOCKS_Y,         ba.size());
    ASSERT_EQ(NXB*N_BLOCKS_X * NYB*N_BLOCKS_Y, ba.numPts());
    for (unsigned int i=0; i<ba.size(); ++i) {
        ASSERT_EQ(ba[i].size(), amrex::IntVect(AMREX_D_DECL(NXB, NYB, 1)));
    }
        
    // Setup with Cartesian coordinate and non-periodic BC so that we can set
    // the BC ourselves
    int coordSystem = 0;  // Cartesian
    amrex::RealBox     physicalDomain({AMREX_D_DECL(X_MIN, Y_MIN, 0.0)},
                                      {AMREX_D_DECL(X_MAX, Y_MAX, 0.0)});
    amrex::Geometry    geometry(domain, physicalDomain,
                                coordSystem, {AMREX_D_DECL(0, 0, 0)});

    // The test problem assumes square mesh
    ASSERT_EQ(geometry.CellSize(0), geometry.CellSize(1));
    ASSERT_TRUE(geometry.IsCartesian());
    ASSERT_FALSE(geometry.isAnyPeriodic());
    
    amrex::MultiFab   mfab(ba, dm, N_VARIABLES, N_GUARD);

    ThreadTeam<Tile>  cpu(4, 1, "TestSingleTeam.log");

    initializeData(geometry, mfab);

    try {
        cpu.startTask(ThreadRoutines::computeLaplacianEnergy_cpu, N_THREADS,
                      "Cpu", "LaplacianEnergy");
        for (amrex::MFIter  itor(mfab); itor.isValid(); ++itor) {
            cpu.enqueue(Tile(itor, mfab, geometry));
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::computeLaplacianDensity_cpu, N_THREADS,
                      "Cpu", "LaplacianDensity");
        for (amrex::MFIter  itor(mfab); itor.isValid(); ++itor) {
            cpu.enqueue(Tile(itor, mfab, geometry));
        }
        cpu.closeTask();
        cpu.wait();

        cpu.startTask(ThreadRoutines::scaleEnergy_cpu, N_THREADS,
                      "Cpu", "scaleEnergy");
        for (amrex::MFIter  itor(mfab); itor.isValid(); ++itor) {
            cpu.enqueue(Tile(itor, mfab, geometry));
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

    amrex::Real    L_inf1      = 0.0;
    amrex::Real    meanAbsErr1 = 0.0;
    amrex::Real    L_inf2      = 0.0;
    amrex::Real    meanAbsErr2 = 0.0;
    computeError(geometry, mfab, &L_inf1, &meanAbsErr1, &L_inf2, &meanAbsErr2);

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

#ifndef VERBOSE
TEST_F(TestRuntimeTile, TestRuntimeSingle) {
    constexpr unsigned int  N_BLOCKS_X     = 512;
    constexpr unsigned int  N_BLOCKS_Y     = 256;

    //***** SETUP DOMAIN, PROBLEM, and MESH
    amrex::IndexType    ccIndexSpace(amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
    amrex::IntVect      domainLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect      domainHi(AMREX_D_DECL(N_BLOCKS_X * NXB - 1,
                                              N_BLOCKS_Y * NYB - 1,
                                              0));
    amrex::Box          domain(domainLo, domainHi, ccIndexSpace);
    amrex::BoxArray     ba(domain);
    ba.maxSize(amrex::IntVect(AMREX_D_DECL(NXB, NYB, 1)));
    amrex::DistributionMapping  dm(ba);

    ASSERT_EQ(N_BLOCKS_X * N_BLOCKS_Y,         ba.size());
    ASSERT_EQ(NXB*N_BLOCKS_X * NYB*N_BLOCKS_Y, ba.numPts());
    for (unsigned int i=0; i<ba.size(); ++i) {
        ASSERT_EQ(ba[i].size(), amrex::IntVect(AMREX_D_DECL(NXB, NYB, 1)));
    }
        
    // Setup with Cartesian coordinate and non-periodic BC so that we can set
    // the BC ourselves
    int coordSystem = 0;  // Cartesian
    amrex::RealBox     physicalDomain({AMREX_D_DECL(X_MIN, Y_MIN, 0.0)},
                                      {AMREX_D_DECL(X_MAX, Y_MAX, 0.0)});
    amrex::Geometry    geometry(domain, physicalDomain,
                                coordSystem, {AMREX_D_DECL(0, 0, 0)});

    // The test problem assumes square mesh
    ASSERT_EQ(geometry.CellSize(0), geometry.CellSize(1));
    ASSERT_TRUE(geometry.IsCartesian());
    ASSERT_FALSE(geometry.isAnyPeriodic());
    
    amrex::MultiFab   mfab(ba, dm, N_VARIABLES, N_GUARD);
    initializeData(geometry, mfab);

    try {
        // Give an extra thread to the GPU task so that it can start to get work
        // to the postGpu task quicker.
        runtime_->executeTask(mfab, geometry,
                              "Task Bundle 1",
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
    computeError(geometry, mfab, &L_inf1, &meanAbsErr1, &L_inf2, &meanAbsErr2);

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
TEST_F(TestRuntimeTile, TestRuntimeScaling) {
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
        //***** SETUP DOMAIN, PROBLEM, and MESH
        amrex::IndexType    ccIndexSpace(amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
        amrex::IntVect      domainLo(AMREX_D_DECL(0, 0, 0));
        amrex::IntVect      domainHi(AMREX_D_DECL(N_BLOCKS_X_ALL[i] * NXB - 1,
                                                  N_BLOCKS_Y_ALL[i] * NYB - 1,
                                                  0));
        amrex::Box          domain(domainLo, domainHi, ccIndexSpace);
        amrex::BoxArray     ba(domain);
        ba.maxSize(amrex::IntVect(AMREX_D_DECL(NXB, NYB, 1)));
        amrex::DistributionMapping  dm(ba);

        ASSERT_EQ(N_BLOCKS_X_ALL[i] * N_BLOCKS_Y_ALL[i],         ba.size());
        ASSERT_EQ(NXB*N_BLOCKS_X_ALL[i] * NYB*N_BLOCKS_Y_ALL[i], ba.numPts());
        for (unsigned int i=0; i<ba.size(); ++i) {
            ASSERT_EQ(ba[i].size(), amrex::IntVect(AMREX_D_DECL(NXB, NYB, 1)));
        }
            
        // Setup with Cartesian coordinate and non-periodic BC so that we can set
        // the BC ourselves
        int coordSystem = 0;  // Cartesian
        amrex::RealBox     physicalDomain({AMREX_D_DECL(X_MIN, Y_MIN, 0.0)},
                                          {AMREX_D_DECL(X_MAX, Y_MAX, 0.0)});
        amrex::Geometry    geometry(domain, physicalDomain,
                                    coordSystem, {AMREX_D_DECL(0, 0, 0)});

        // The test problem assumes square mesh
        ASSERT_EQ(geometry.CellSize(0), geometry.CellSize(1));
        ASSERT_TRUE(geometry.IsCartesian());
        ASSERT_FALSE(geometry.isAnyPeriodic());
        
        amrex::MultiFab   mfab(ba, dm, N_VARIABLES, N_GUARD);
        initializeData(geometry, mfab);

        //***** COMPUTATION STAGE
        // At this level, the Orchestration System has been used to contruct the
        // operation function that we call.  Hopefully, the interface of this
        // function will not vary with the Orchestration Runtime chosen at setup.
        // So far, the Driver is Runtime agnostic.
        try {
            // myGrid is a standin for the set of parameters we need to
            // specify the tile iterator to use.
            runtime_->executeTask(mfab, geometry,
                                  "Task Bundle 1",
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

        computeError(geometry, mfab, &L_inf1, &meanAbsErr1, &L_inf2, &meanAbsErr2);

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
