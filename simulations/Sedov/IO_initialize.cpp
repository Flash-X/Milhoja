#include "IO.h"
#include "Orchestration.h"

#include <cstdio>
#include <stdexcept>

#include <mpi.h>

#include "Grid.h"

#include "Flash.h"

/**
 * Initialize the IO unit.  This should only be called after the Grid unit has
 * been initialized and has set up the mesh in accord with the initial
 * conditions.
 *
 * \todo The blockIntegralQuantities_* arrays should not be allocated here.
 * Rather, regridding should allocate them based on the current number of blocks
 * managed by the process.  In this case, we should be able to call this before
 * initDomain.
 *
 * \todo Include the possibility of starting a simulation as a restart.  Look at
 * FLASH-X to see how they manange files like the integral quantities output.
 *
 * \todo Allow for the possibility of writing out mass scalar integral
 * quantities.
 *
 * \param  filename - the name of file to which integral quantities will be
 *                    written.
 */
void   IO::initialize(const std::string filename) {
    using namespace orchestration;

    // Number of globally-summed regular quantities
#ifdef MAGP_VAR_C
    constexpr  int  N_GLOBAL_SUM_PROP = 8;
#else
    constexpr  int  N_GLOBAL_SUM_PROP = 7;
#endif
    constexpr  int  N_GLOBAL_SUM = N_GLOBAL_SUM_PROP + NMASS_SCALARS;

    int  rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nIntegralQuantities = N_GLOBAL_SUM_PROP;
//    if (io_writeMscalarIntegrals) {
//       nGlobalSumUsed = nGlobalSum
//    } else {
//       nGlobalSumUsed = nGlobalSumProp
//    }
    localIntegralQuantities  = new Real[nIntegralQuantities];
    if (rank == MASTER_PE) {
        globalIntegralQuantities = new Real[nIntegralQuantities];
    }

    if (filename == "") {
        throw std::invalid_argument("[IO::initalize] Empty fileaname given");
    }
    io::integralQuantitiesFilename = filename;

    // TODO: We can't have namespaces
    //  orchestration, orch, and Orchestration.  Do better.
    unsigned int   nThreads = Orchestration::nThreadsPerTeam;
#ifdef DENS_VAR_C
    io::blockIntegralQuantities_mass = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_mass[i] = 0.0;
    }
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
    io::blockIntegralQuantities_xmom = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_xmom[i] = 0.0;
    }
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
    io::blockIntegralQuantities_ymom = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_ymom[i] = 0.0;
    }
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
    io::blockIntegralQuantities_zmom = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_zmom[i] = 0.0;
    }
#endif
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
    io::blockIntegralQuantities_ener = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_ener[i] = 0.0;
    }
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
    io::blockIntegralQuantities_ke   = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_ke[i] = 0.0;
    }
#endif
#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
    io::blockIntegralQuantities_eint = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_eint[i] = 0.0;
    }
#endif
#ifdef MAGP_VAR_C
    io::blockIntegralQuantities_magp = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        io::blockIntegralQuantities_magp[i] = 0.0;
    }
#endif

    if (rank == MASTER_PE) {
        FILE*   fptr = fopen(io::integralQuantitiesFilename.c_str(), "a");
        if (!fptr) {
            std::string msg = "[IO::initalize] Unable to open integral quantities output file "
                              + filename;
            throw std::runtime_error(msg);
        }

#ifdef MAGP_VAR_C
        fprintf(fptr, "#%24s %25s %25s %25s %25s %25s %25s %25s %25s\n",
                    "time", 
                    "mass",
                    "x-momentum",
                    "y-momentum",
                    "z-momentum",
                    "E_total",
                    "E_kinetic",
                    "E_internal",
                    "MagEnergy");
#else
        fprintf(fptr, "#%24s %25s %25s %25s %25s %25s %25s %25s\n",
                    "time", 
                    "mass",
                    "x-momentum",
                    "y-momentum",
                    "z-momentum",
                    "E_total",
                    "E_kinetic",
                    "E_internal");
#endif
        fclose(fptr);
        fptr = NULL;
    }
}

