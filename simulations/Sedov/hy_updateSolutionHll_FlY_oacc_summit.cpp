#include "Hydro.h"

#include "Flash.h"

void hy::updateSolutionHll_FlY_oacc_summit(const orchestration::IntVect& lo,
                                           const orchestration::IntVect& hi,
                                           orchestration::FArray4D& U,
                                           const orchestration::FArray4D& flY) {
    using namespace orchestration;

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, DENS_VAR_C) += (  flY(i, j,   k, HY_DENS_FLUX_C)
                                           - flY(i, j+1, k, HY_DENS_FLUX_C) );
                U(i, j, k, VELX_VAR_C) += (  flY(i, j,   k, HY_XMOM_FLUX_C)
                                           - flY(i, j+1, k, HY_XMOM_FLUX_C) );
                U(i, j, k, VELY_VAR_C) += (  flY(i, j,   k, HY_YMOM_FLUX_C)
                                           - flY(i, j+1, k, HY_YMOM_FLUX_C) );
                U(i, j, k, VELZ_VAR_C) += (  flY(i, j,   k, HY_ZMOM_FLUX_C)
                                           - flY(i, j+1, k, HY_ZMOM_FLUX_C) );
                U(i, j, k, ENER_VAR_C) += (  flY(i, j,   k, HY_ENER_FLUX_C)
                                           - flY(i, j+1, k, HY_ENER_FLUX_C) );
            }
        }
    }
}

