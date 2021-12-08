#ifndef MILHOJA_H__
#define MILHOJA_H__

#if 0
#####----- FIXED CONSTANTS
#endif
#define MDIM 3

#if 0
#####----- CONSTANTS PASSED TO BUILD SYSTEM
#endif

#define REAL_IS_DOUBLE

#ifdef REAL_IS_DOUBLE
#define MILHOJA_MPI_REAL      MPI_DOUBLE_PRECISION
#else
#error "Invalid real type"
#endif

#if 0
# This should eventually be set automatically by the build system based on the
# build-time parameter that specifies the desired Grid backend.
#endif
#define GRID_AMREX

#endif

