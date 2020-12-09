#include "Hydro.h"

#include "Flash.h"

void hy::scaleSolutionHll_oacc_summit(const orchestration::IntVect& lo,
                                      const orchestration::IntVect& hi,
                                      const orchestration::FArray4D& Uin,
                                      orchestration::FArray4D& Uout) {
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
                Uout(i, j, k, DENS_VAR_C) =   Uin(i,   j, k, DENS_VAR_C);
            }
        }
    }
}

