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

#include "scale_cpu.h"
#include "computeLaplacian_cpu.h"

// TASK_COMPOSER: The offline tool will need to determine how many thread teams
// are needed as well as how many threads to allocate to each.
const unsigned int N_THREAD_TEAMS = 3;
const unsigned int MAX_THREADS = 10;

#if defined(P1)
//***** PROBLEM ONE
//      Approximated exactly by second-order discretized Laplacian
double f(const double x, const double y) {
    return (  3.0*x*x*x +     x*x + x 
            - 2.0*y*y*y - 1.5*y*y + y
            + 5.0);
}

double Delta_f(const double x, const double y) {
    return (18.0*x - 12.0*y - 1.0);
}
#elif defined(P2)
//***** PROBLEM TWO
//      Approximation is not exact and we know the error term exactly
double f(const double x, const double y) {
    return (  4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
            -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
            + 1.0);
}

double Delta_f(const double x, const double y) {
    return (  48.0*x*x - 18.0*x
            - 12.0*y*y + 12.0*y
            - 2.0); 
}
#endif

void initializeData(BlockIterator& itor) {
    int      i0   = 0;
    int      j0   = 0;
    double   x    = 0.0;
    double** data = NULL;

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
                   data[i-i0][j-j0] = f(x, y_coords[j-j0]);
              }
         }
    }
}

void computeError(BlockIterator& itor, double* L_inf, double* meanAbsErr) {
    double**                      data = NULL;
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
    double maxAbsErr    = 0.0;
    double sum          = 0.0;
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
                   absErr = fabs(2.1 * Delta_f(x, y) - data[i-i0][j-j0]);
                   
                   sum += absErr;
                   
                   if (absErr > maxAbsErr) {
                        maxAbsErr = absErr;
                   }

                   ++nCells;
              }
         }
    }

    *L_inf = maxAbsErr;
    *meanAbsErr = sum / ((double)nCells);
}

int main(int argc, char* argv[]) {
    //***** PROBLEM DEFINITION/CONFIGURATION
    // Define problem and operation/infrastructure configuration
    // Maintain dx=dy at each trial for simple scaling test
#if   defined(SINGLE)
    const unsigned int N_TRIALS = 1;
    const unsigned int N_BLOCKS_X_ALL[N_TRIALS] = {6};
    const unsigned int N_BLOCKS_Y_ALL[N_TRIALS] = {3};
#elif defined(SCALING)
    const unsigned int N_TRIALS = 6;
    const unsigned int N_BLOCKS_X_ALL[N_TRIALS] = {2, 4, 6, 8, 10, 12};
    const unsigned int N_BLOCKS_Y_ALL[N_TRIALS] = {1, 2, 3, 4,  5,  6};
//    const unsigned int N_TRIALS = 8;
//    const unsigned int N_BLOCKS_X_ALL[N_TRIALS] = {8, 16, 32, 64, 128, 256, 512, 1024};
//    const unsigned int N_BLOCKS_Y_ALL[N_TRIALS] = {4,  8, 16, 32,  64, 128, 256,  512};
#endif

    const double       X_MIN      = 0.0;
    const double       X_MAX      = 1.0;
    const double       Y_MIN      = 0.0;
    const double       Y_MAX      = 1.0;
    const unsigned int N_GUARD    = 1;
    const unsigned int NXB        = 8;
    const unsigned int NYB        = 16;
    const unsigned int MAX_BLOCKS = 100;

    //***** DATA COLLECTION
    unsigned int                  nGuard = 0;
    std::array<double,NDIM>       domainLo;
    std::array<double,NDIM>       domainHi;
    std::array<unsigned int,NDIM> domainShape;
    std::array<unsigned int,NDIM> blockSize;
    std::array<double,NDIM>       deltas;

    OrchestrationRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    OrchestrationRuntime::setMaxThreadsPerTeam(MAX_THREADS);
    OrchestrationRuntime*    runtime = OrchestrationRuntime::instance();

    double L_inf      = 0.0;
    double meanAbsErr = 0.0;
    double results[N_TRIALS][3];
    for (int i=0; i<N_TRIALS; ++i) {
        //***** SETUP DOMAIN MESH
        Grid myGrid(X_MIN, X_MAX, Y_MIN, Y_MAX,
                    NXB, NYB, N_BLOCKS_X_ALL[i], N_BLOCKS_Y_ALL[i],
                    N_GUARD);

        // Use getters to test Grid class
        nGuard      = myGrid.nGuardcells();
        domainLo    = myGrid.domain(LOW);
        domainHi    = myGrid.domain(HIGH);
        domainShape = myGrid.shape();
        blockSize   = myGrid.blockSize();
        deltas      = myGrid.deltas();

        assert(deltas[0] == deltas[1]);

        printf("\n");
        printf("Domain Lo\t\t(%f, %f)\n", domainLo[IAXIS], domainLo[JAXIS]);
        printf("Domain Hi\t\t(%f, %f)\n", domainHi[IAXIS], domainHi[JAXIS]);
        printf("Domain Block Shape\t%d x %d\n", domainShape[IAXIS], domainShape[JAXIS]);
        printf("Deltas\t\t\t(%f, %f)\n", deltas[IAXIS], deltas[JAXIS]);
        printf("Block Size\t\t%d x %d\n", blockSize[IAXIS], blockSize[JAXIS]);
        printf("Number of Guardcells\t%d\n", nGuard);
        printf("\n");

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
            runtime->executeTask(myGrid, "Task Bundle 1",
                                 ThreadRoutines::computeLaplacian_cpu, 2, "bundle1_cpuTask",
                                 ThreadRoutines::computeLaplacian_cpu, 5, "bundle1_gpuTask",
                                 ThreadRoutines::scale_cpu, 0, "bundle1_postGpuTask");
        } catch (std::invalid_argument  e) {
            printf("\nINVALID ARGUMENT: %s\n\n", e.what());
            return 2;
        } catch (std::logic_error  e) {
            printf("\nLOGIC ERROR: %s\n\n", e.what());
            return 3;
        } catch (std::runtime_error  e) {
            printf("\nRUNTIME ERROR: %s\n\n", e.what());
            return 4;
        } catch (...) {
            printf("\n??? ERROR: Unanticipated error\n\n");
            return 5;
        }

        computeError(itor, &L_inf, &meanAbsErr);
        results[i][0] = NXB * N_BLOCKS_X_ALL[i];
        results[i][1] = L_inf;
        results[i][2] = meanAbsErr;

        // Write to file the finest solution
//        if (i == (N_TRIALS - 1)) {
//            FILE* fp = fopen("laplacian.dat", "w");
//
//            for (itor.clear(); itor.isValid(); itor.next()) {
//                Block block = itor.currentBlock();
//
//                double**                       dataPtr   = block.dataPtr();
//                std::array<unsigned int, NDIM> lo        = block.lo();
//                std::array<unsigned int, NDIM> hi        = block.hi();
//                std::array<int, NDIM>          loGC      = block.loGC();
//                std::vector<double>            xCoords   = block.coordinates(IAXIS);
//                std::vector<double>            yCoords   = block.coordinates(JAXIS);
//
//                unsigned int i0 = loGC[IAXIS];
//                unsigned int j0 = loGC[JAXIS];
//                for     (unsigned int i2=lo[IAXIS]; i2<=hi[IAXIS]; ++i2) {
//                    for (unsigned int j2=lo[JAXIS]; j2<=hi[JAXIS]; ++j2) {
//                        fprintf(fp, "%d\t%d\t%.16f\t%.16f\t%.16f\n", i2, j2, 
//                                    xCoords[i2-i0], yCoords[j2-j0],
//                                    dataPtr[i2-i0][j2-j0]);
//                    }
//                }
//            }
//
//            fclose(fp);
//        }
    }
    printf("\n");

    //***** WRITE RESULTS FOR VERIFICATION
    unsigned int nCells     = 0;
    double       slope      = 0.0;
    double       deltaErr   = 0.0;
    double       deltaCells = 0.0;

    printf("nCells\tL_inf\t\tMean\t\tMean Slope\n");
    for (int i=0; i<N_TRIALS; ++i) {
        nCells     = (int)results[i][0];
        L_inf      =      results[i][1];
        meanAbsErr =      results[i][2];
        if        (i == 0) {
            printf("%d\t%g\t%g\t      -\n", nCells, L_inf, meanAbsErr);
        } else if (meanAbsErr == 0.0) {
            printf("%d\t%g\t%g\tn/a\n", nCells, L_inf, meanAbsErr);
        } else {
            deltaErr   = log10(results[i][2]) -log10(results[i-1][2]);
            deltaCells = log10(results[i][0]) -log10(results[i-1][0]);
            slope = deltaErr / deltaCells;
            printf("%d\t%g\t%g\t%.8f\n", nCells, L_inf, meanAbsErr, slope);
        }
    }

    // TODO: Add test to confirm reasonably small mean error
    delete OrchestrationRuntime::instance();

    pthread_exit(NULL);

    return 0;
}

