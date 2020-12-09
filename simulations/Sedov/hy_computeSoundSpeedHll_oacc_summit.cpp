#include "Hydro.h"

#include <cmath>

#include "Flash.h"

void hy::computeSoundSpeedHll_oacc_summit(const orchestration::IntVect& lo,
                                          const orchestration::IntVect& hi,
                                          const orchestration::FArray4D& U,
                                          orchestration::FArray3D& auxC) {
    using namespace orchestration;

    for         (int k=lo.K()-K3D; k<=hi.K()+K3D; ++k) {
        for     (int j=lo.J()-K2D; j<=hi.J()+K2D; ++j) {
            for (int i=lo.I()-K1D; i<=hi.I()+K1D; ++i) {
                auxC(i, j, k) = sqrt(  U(i, j, k, GAMC_VAR_C)
                                     * U(i, j, k, PRES_VAR_C)
                                     / U(i, j, k, DENS_VAR_C) );
            }
        }
    }
}

