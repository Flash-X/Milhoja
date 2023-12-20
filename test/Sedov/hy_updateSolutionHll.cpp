#include "Hydro.h"

#include "Sedov.h"

void hy::updateSolutionHll(const milhoja::IntVect& lo,
                           const milhoja::IntVect& hi,
                           milhoja::FArray4D& U,
                           LIST_NDIM(const milhoja::FArray4D& flX, const milhoja::FArray4D& flY, const milhoja::FArray4D& flZ)) {
    //$milhoja  "flX": {
    //$milhoja&   "R": [HY_XMOM_FLUX, HY_YMOM_FLUX, HY_ZMOM_FLUX,
    //$milhoja&         HY_DENS_FLUX, HY_ENER_FLUX]
    //$milhoja& },
    //$milhoja& "flY": {
    //$milhoja&   "R": [HY_XMOM_FLUX, HY_YMOM_FLUX, HY_ZMOM_FLUX,
    //$milhoja&         HY_DENS_FLUX, HY_ENER_FLUX]
    //$milhoja& },
    //$milhoja& "flZ": {
    //$milhoja&   "R": [HY_XMOM_FLUX, HY_YMOM_FLUX, HY_ZMOM_FLUX,
    //$milhoja&         HY_DENS_FLUX, HY_ENER_FLUX]
    //$milhoja& },
    //$milhoja& "U": {
    //$milhoja&   "RW": [VELX_VAR, VELY_VAR, VELZ_VAR, DENS_VAR, ENER_VAR],
    //$milhoja&   "W":  [EINT_VAR]
    //$milhoja& }

    using namespace milhoja;

#ifdef EINT_VAR
    Real    norm2_sqr = 0.0_wp;
#endif
    Real    densOld = 0.0_wp;
    Real    densNew = 0.0_wp;
    Real    densNew_inv = 0.0_wp;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                // Update density first
                densOld = U(i, j, k, DENS_VAR);
#if   MILHOJA_NDIM == 1
                densNew =   densOld
                          + flX(i,   j, k, HY_DENS_FLUX)
                          - flX(i+1, j, k, HY_DENS_FLUX);
#elif MILHOJA_NDIM == 2
                densNew =   densOld
                          + flX(i,   j,   k, HY_DENS_FLUX)
                          - flX(i+1, j,   k, HY_DENS_FLUX)
                          + flY(i,   j,   k, HY_DENS_FLUX)
                          - flY(i,   j+1, k, HY_DENS_FLUX);
#elif MILHOJA_NDIM == 3
                densNew =   densOld 
                          + flX(i,   j,   k,   HY_DENS_FLUX)
                          - flX(i+1, j,   k,   HY_DENS_FLUX)
                          + flY(i,   j,   k,   HY_DENS_FLUX)
                          - flY(i,   j+1, k,   HY_DENS_FLUX)
                          + flZ(i,   j,   k,   HY_DENS_FLUX)
                          - flZ(i,   j,   k+1, HY_DENS_FLUX);
#endif
                U(i, j, k, DENS_VAR) = densNew;
                densNew_inv = 1.0_wp / densNew;

                // velocities and total energy can be updated independently
                // using density result
#if   MILHOJA_NDIM == 1
                U(i, j, k, VELX_VAR) = (  U(i, j, k, VELX_VAR) * densOld
                                        + flX(i,   j, k, HY_XMOM_FLUX)
                                        - flX(i+1, j, k, HY_XMOM_FLUX) ) * densNew_inv;

                U(i, j, k, VELY_VAR) = (  U(i, j, k, VELY_VAR) * densOld
                                        + flX(i,   j, k, HY_YMOM_FLUX)
                                        - flX(i+1, j, k, HY_YMOM_FLUX) ) * densNew_inv;

                U(i, j, k, VELZ_VAR) = (  U(i, j, k, VELZ_VAR) * densOld
                                        + flX(i,   j, k, HY_ZMOM_FLUX)
                                        - flX(i+1, j, k, HY_ZMOM_FLUX) ) * densNew_inv;

                U(i, j, k, ENER_VAR) = (  U(i, j, k, ENER_VAR) * densOld
                                        + flX(i,   j, k, HY_ENER_FLUX)
                                        - flX(i+1, j, k, HY_ENER_FLUX) ) * densNew_inv;
#elif MILHOJA_NDIM == 2
                U(i, j, k, VELX_VAR) = (  U(i, j, k, VELX_VAR) * densOld
                                        + flX(i,   j,   k, HY_XMOM_FLUX)
                                        - flX(i+1, j,   k, HY_XMOM_FLUX)
                                        + flY(i,   j,   k, HY_XMOM_FLUX)
                                        - flY(i,   j+1, k, HY_XMOM_FLUX) ) * densNew_inv;

                U(i, j, k, VELY_VAR) = (  U(i, j, k, VELY_VAR) * densOld
                                        + flX(i,   j,   k, HY_YMOM_FLUX)
                                        - flX(i+1, j,   k, HY_YMOM_FLUX)
                                        + flY(i,   j,   k, HY_YMOM_FLUX)
                                        - flY(i,   j+1, k, HY_YMOM_FLUX) ) * densNew_inv;

                U(i, j, k, VELZ_VAR) = (  U(i, j, k, VELZ_VAR) * densOld
                                        + flX(i,   j,   k, HY_ZMOM_FLUX)
                                        - flX(i+1, j,   k, HY_ZMOM_FLUX)
                                        + flY(i,   j,   k, HY_ZMOM_FLUX)
                                        - flY(i,   j+1, k, HY_ZMOM_FLUX) ) * densNew_inv;

                U(i, j, k, ENER_VAR) = (  U(i, j, k, ENER_VAR) * densOld
                                        + flX(i,   j,   k, HY_ENER_FLUX)
                                        - flX(i+1, j,   k, HY_ENER_FLUX)
                                        + flY(i,   j,   k, HY_ENER_FLUX)
                                        - flY(i,   j+1, k, HY_ENER_FLUX) ) * densNew_inv;
#elif MILHOJA_NDIM == 3
                U(i, j, k, VELX_VAR) = (  U(i, j, k, VELX_VAR) * densOld
                                        + flX(i,   j,   k,   HY_XMOM_FLUX)
                                        - flX(i+1, j,   k,   HY_XMOM_FLUX)
                                        + flY(i,   j,   k,   HY_XMOM_FLUX)
                                        - flY(i,   j+1, k,   HY_XMOM_FLUX)
                                        + flZ(i,   j,   k,   HY_XMOM_FLUX)
                                        - flZ(i,   j,   k+1, HY_XMOM_FLUX) ) * densNew_inv;

                U(i, j, k, VELY_VAR) = (  U(i, j, k, VELY_VAR) * densOld
                                        + flX(i,   j,   k,   HY_YMOM_FLUX)
                                        - flX(i+1, j,   k,   HY_YMOM_FLUX)
                                        + flY(i,   j,   k,   HY_YMOM_FLUX)
                                        - flY(i,   j+1, k,   HY_YMOM_FLUX)
                                        + flZ(i,   j,   k,   HY_YMOM_FLUX)
                                        - flZ(i,   j,   k+1, HY_YMOM_FLUX) ) * densNew_inv;

                U(i, j, k, VELZ_VAR) = (  U(i, j, k, VELZ_VAR) * densOld
                                        + flX(i,   j,   k,   HY_ZMOM_FLUX)
                                        - flX(i+1, j,   k,   HY_ZMOM_FLUX)
                                        + flY(i,   j,   k,   HY_ZMOM_FLUX)
                                        - flY(i,   j+1, k,   HY_ZMOM_FLUX)
                                        + flZ(i,   j,   k,   HY_ZMOM_FLUX)
                                        - flZ(i,   j,   k+1, HY_ZMOM_FLUX) ) * densNew_inv;

                U(i, j, k, ENER_VAR) = (  U(i, j, k, ENER_VAR) * densOld
                                        + flX(i,   j,   k,   HY_ENER_FLUX)
                                        - flX(i+1, j,   k,   HY_ENER_FLUX)
                                        + flY(i,   j,   k,   HY_ENER_FLUX)
                                        - flY(i,   j+1, k,   HY_ENER_FLUX)
                                        + flZ(i,   j,   k,   HY_ENER_FLUX)
                                        - flZ(i,   j,   k+1, HY_ENER_FLUX) ) * densNew_inv;
#endif

#ifdef EINT_VAR
                // Compute energy correction from new velocities and energy
                norm2_sqr =   U(i, j, k, VELX_VAR) * U(i, j, k, VELX_VAR)
                            + U(i, j, k, VELY_VAR) * U(i, j, k, VELY_VAR)
                            + U(i, j, k, VELZ_VAR) * U(i, j, k, VELZ_VAR);
                U(i, j, k, EINT_VAR) =    U(i, j, k, ENER_VAR)
                                         - (0.5_wp * norm2_sqr);
#endif
            }
        }
    }
}

