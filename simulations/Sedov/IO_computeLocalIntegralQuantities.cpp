#include "IO.h"
#include "Orchestration.h"

#include "Grid.h"

/**
 * Use the integral quantities as pre-computed on a per-block basis to compute
 * the same quantities integrated across all blocks managed by the process.
 *
 * The results are stored in the IO unit's localIntegralQuantities array.
 *
 * This function leaves all blockIntegralQuantities_* buffers zeroed so that
 * they are ready for immediate use the next time that integral quantities need
 * to be computed.
 */
void   IO::computeLocalIntegralQuantities(void) {
    for (unsigned int i=0; i<nIntegralQuantities; ++i) {
        localIntegralQuantities[i] = 0.0_wp;
    }

    // Any or all of the threads in the thread team that ran
    // computeBlockIntegralQuantities could have computed part of the
    // integration.  Therefore, we integrate across all threads.
    for (unsigned int i=0; i<Orchestration::nThreadsPerTeam; ++i) {
        // The order in which quantities are stored in the array must match the
        // order in which quantities are listed in the output file's header.
#ifdef DENS_VAR_C
        localIntegralQuantities[0] += io::blockIntegralQuantities_mass[i];
        io::blockIntegralQuantities_mass[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
        localIntegralQuantities[1] += io::blockIntegralQuantities_xmom[i];
        io::blockIntegralQuantities_xmom[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
        localIntegralQuantities[2] += io::blockIntegralQuantities_ymom[i];
        io::blockIntegralQuantities_ymom[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
        localIntegralQuantities[3] += io::blockIntegralQuantities_zmom[i];
        io::blockIntegralQuantities_zmom[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
        localIntegralQuantities[4] += io::blockIntegralQuantities_ener[i];
        io::blockIntegralQuantities_ener[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
        localIntegralQuantities[5] += io::blockIntegralQuantities_ke[i];
        io::blockIntegralQuantities_ke[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
        localIntegralQuantities[6] += io::blockIntegralQuantities_eint[i];
        io::blockIntegralQuantities_eint[i] = 0.0_wp;
#endif
#ifdef MAGP_VAR_C
        localIntegralQuantities[7] += io::blockIntegralQuantities_magp[i];
        io::blockIntegralQuantities_magp[i] = 0.0_wp;
#endif
    }
}

