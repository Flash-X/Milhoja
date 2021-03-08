#include "estimateTimerResolution.h"

//#include <mpi.h>
#include <limits>
#include <chrono>
#include <iostream>
#include <stdexcept>

unsigned int  foo(const unsigned int n) {
    unsigned int i = 0;
    for (i=0; i<n; ++i) {}
    return i;
}

//double estimateMpiTimerResolution(void) {
//    constexpr unsigned int   N_TRIALS = 1000;
//    
//    unsigned int   n = 0;
//    double         t0 = 0.0;
//    double         walltime = 0.0;
//    double         min_walltime = 1.0e100;
//    unsigned int   max_iters = 0;
//    for (unsigned int i=0; i<N_TRIALS; ++i) {
//        n = 0;
//        do {
//            t0 = MPI_Wtime();
//            foo(n);
//            walltime = MPI_Wtime() - t0;
//            ++n;
//        } while (walltime == 0.0);
//
//        if (walltime < min_walltime) {
//            min_walltime = walltime;
//        }
//        if (n > max_iters) {
//            max_iters = n;
//        }
//    }
//
//    if (max_iters <= 1) {
//        throw std::runtime_error("Unable to estimate MPI_Wtime resolution");
//    }
//
//    return min_walltime;
//}

double estimateSteadyClockResolution(void) {
    using microseconds = std::chrono::duration<double, std::micro>;

    constexpr unsigned int   N_TRIALS = 1000;
    
    unsigned int   n = 0;
    auto           start = std::chrono::steady_clock::now();
    auto           end   = std::chrono::steady_clock::now();
    double         walltime = 0.0;
    double         min_walltime = std::numeric_limits<double>::max();
    unsigned int   max_iters = 0;
    for (unsigned int i=0; i<N_TRIALS; ++i) {
        n = 0;
        do {
            start = std::chrono::steady_clock::now();
            foo(n);
            end = std::chrono::steady_clock::now();

            walltime = microseconds(end - start).count();
            ++n;
        } while (walltime == 0.0);

        if (walltime < min_walltime) {
            min_walltime = walltime;
        }
        if (n > max_iters) {
            max_iters = n;
        }
    }

    if (max_iters <= 1) {
        throw std::runtime_error("Unable to estimate steady_clock resolution");
    }

    return min_walltime;
}

