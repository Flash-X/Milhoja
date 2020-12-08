#include "IO.h"

/**
 * Integrate the physical integral quantities of interest across the given
 * region.  Note that this function
 *  - is not intended be applied to any arbitrary rectangular region of
 *    the domain, but rather to blocks,
 *  - is intended for computation across the interior of the given block, and
 *  - is intended for use with leaf blocks.
 * 
 * The integrated quantities for the given block are accumulated into a given
 * array element.  It is assumed that the management of array elements is such
 * that the thread that runs this function has exclusive access to its given
 * array element.
 * 
 * \param simTime - the time at which the given data is the valid solution
 * \param idx - the index of the array index into which this result should
 *              accumulate
 * \param lo - the lo corner of the block's interior
 * \param hi - the hi corner of the block's interior
 * \param cellVolumes - the volumes of each cell in the block
 * \param solnData - the data to integrate
 */
void   IO::computeBlockIntegralQuantities(const orchestration::Real simTime,
                                          const int idx,
                                          const orchestration::IntVect& lo,
                                          const orchestration::IntVect& hi,
                                          const orchestration::FArray3D& cellVolumes,
                                          const orchestration::FArray4D& solnData) {
    using namespace orchestration;

    Real    dvol = 0.0_wp;
    Real    mass = 0.0_wp;
    Real    massSum = 0.0_wp;
    Real    xmomSum = 0.0_wp;
    Real    ymomSum = 0.0_wp;
    Real    zmomSum = 0.0_wp;
    Real    enerSum = 0.0_wp;
    Real    keSum   = 0.0_wp;
    Real    eintSum = 0.0_wp;
    Real    magpSum = 0.0_wp;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                dvol = cellVolumes(i, j, k);

#if defined(DENS_VAR_C)
                // mass
                mass = solnData(i, j, k, DENS_VAR_C) * dvol;
                massSum += mass;
#endif

                // momentum
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
                xmomSum += mass * solnData(i, j, k, VELX_VAR_C);
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
                ymomSum += mass * solnData(i, j, k, VELY_VAR_C);
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
                zmomSum += mass * solnData(i, j, k, VELZ_VAR_C);
#endif

                // total energy
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
                enerSum += mass * solnData(i, j, k, ENER_VAR_C);
#ifdef MAGP_VAR_C
                // total plasma energy
                enerSum += solnData(i, j, k, MAGP_VAR_C) * dvol;
#endif
#endif

#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
                // kinetic energy
                keSum += 0.5_wp * mass
                                * (  std::pow(solnData(i, j, k, VELX_VAR_C), 2)
                                   + std::pow(solnData(i, j, k, VELY_VAR_C), 2)
                                   + std::pow(solnData(i, j, k, VELZ_VAR_C), 2));
#endif

#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
              // internal energy
              eintSum += mass * solnData(i, j, k, EINT_VAR_C);
#endif

#ifdef MAGP_VAR_C
              // magnetic energy
              magpSum += solnData(i, j, k, MAGP_VAR_C) * dvol;
#endif

              // TODO: Sum mass scalar quantities
            }
        }
    }

#ifdef DENS_VAR_C
    io::blockIntegralQuantities_mass[idx] += massSum;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
    io::blockIntegralQuantities_xmom[idx] += xmomSum;
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
    io::blockIntegralQuantities_ymom[idx] += ymomSum;
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
    io::blockIntegralQuantities_zmom[idx] += zmomSum;
#endif
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
    io::blockIntegralQuantities_ener[idx] += enerSum;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
    io::blockIntegralQuantities_ke[idx]   += keSum;
#endif
#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
    io::blockIntegralQuantities_eint[idx] += eintSum;
#endif
#ifdef MAGP_VAR_C
    io::blockIntegralQuantities_magp[idx] += magpSum;
#endif
}

