#include "Hydro.h"

#include "Flash.h"

void hy::rescaleSolutionHll_oacc_summit(const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        orchestration::FArray4D& U) {
    using namespace orchestration;

    Real   invNewDens = 0.0_wp;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                invNewDens = 1.0_wp / U(i, j, k, DENS_VAR_C);

                U(i, j, k, VELX_VAR_C) *= invNewDens;
                U(i, j, k, VELY_VAR_C) *= invNewDens;
                U(i, j, k, VELZ_VAR_C) *= invNewDens;
                U(i, j, k, ENER_VAR_C) *= invNewDens;
            }
        }
    }
}

