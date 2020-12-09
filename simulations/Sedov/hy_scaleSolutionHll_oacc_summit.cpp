#include "Hydro.h"

#include "Flash.h"

void hy::scaleSolutionHll_oacc_summit(const orchestration::IntVect& lo,
                                      const orchestration::IntVect& hi,
                                      const orchestration::FArray4D& Uin,
                                      orchestration::FArray4D& Uout,
                                      const orchestration::FArray4D& flX,
                                      const orchestration::FArray4D& flY,
                                      const orchestration::FArray4D& flZ) {
    using namespace orchestration;

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                Uout(i, j, k, VELX_VAR_C) =   Uin(i,   j, k, VELX_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, VELY_VAR_C) =   Uin(i,   j, k, VELY_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, VELZ_VAR_C) =   Uin(i,   j, k, VELZ_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, ENER_VAR_C) =   Uin(i,   j, k, ENER_VAR_C)
                                            * Uin(i,   j, k, DENS_VAR_C);
                Uout(i, j, k, DENS_VAR_C) =   Uin(i,   j, k, DENS_VAR_C)
                                            + flX(i,   j, k, HY_DENS_FLUX_C)
                                            - flX(i+1, j, k, HY_DENS_FLUX_C);
                // TODO: After we get timing data on this code, replace the branches
                // here with preprocessor macros to get an idea of the performance
                // hit of the branch statements.
                if (NDIM > 1) {
                     Uout(i, j, k, DENS_VAR_C) += (  flY(i, j,   k, HY_DENS_FLUX_C)
                                                   - flY(i, j+1, k, HY_DENS_FLUX_C) );
                }
                if (NDIM > 2) {
                     Uout(i, j, k, DENS_VAR_C) += (  flZ(i, j, k,   HY_DENS_FLUX_C)
                                                   - flZ(i, j, k+1, HY_DENS_FLUX_C) );
                }
            }
        }
    }
}

