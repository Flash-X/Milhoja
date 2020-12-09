#include "Hydro.h"

#include "Flash.h"

void hy::updateVelyHll_oacc_summit(const orchestration::IntVect& lo,
                                   const orchestration::IntVect& hi,
                                   const orchestration::FArray4D& Uin,
                                   orchestration::FArray4D& Uout,
                                   const orchestration::FArray4D& flX,
                                   const orchestration::FArray4D& flY,
                                   const orchestration::FArray4D& flZ) {
    using namespace orchestration;

    Real    invNewDens = 0.0_wp;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                invNewDens = 1.0_wp / Uout(i, j, k, DENS_VAR_C);

#if NDIM == 1
                Uout(i, j, k, VELY_VAR_C) = (  Uin(i, j, k, VELY_VAR_C) * Uin(i, j, k, DENS_VAR_C)
                                             + flX(i,   j, k, HY_YMOM_FLUX_C)
                                             - flX(i+1, j, k, HY_YMOM_FLUX_C) ) * invNewDens;
#elif NDIM == 2
                Uout(i, j, k, VELY_VAR_C) = (  Uin(i, j, k, VELY_VAR_C) * Uin(i, j, k, DENS_VAR_C)
                                             + flX(i,   j,   k, HY_YMOM_FLUX_C)
                                             - flX(i+1, j,   k, HY_YMOM_FLUX_C)
                                             + flY(i,   j,   k, HY_YMOM_FLUX_C)
                                             - flY(i,   j+1, k, HY_YMOM_FLUX_C) ) * invNewDens;
#elif NDIM == 3
                Uout(i, j, k, VELY_VAR_C) = (  Uin(i, j, k, VELY_VAR_C) * Uin(i, j, k, DENS_VAR_C)
                                             + flX(i,   j,   k,   HY_YMOM_FLUX_C)
                                             - flX(i+1, j,   k,   HY_YMOM_FLUX_C)
                                             + flY(i,   j,   k,   HY_YMOM_FLUX_C)
                                             - flY(i,   j+1, k,   HY_YMOM_FLUX_C)
                                             + flZ(i,   j,   k,   HY_YMOM_FLUX_C)
                                             - flZ(i,   j,   k+1, HY_YMOM_FLUX_C) ) * invNewDens;
#endif
            }
        }
    }
}

