#include "Eos.h"

#include "Flash.h"

namespace Eos {

void idealGamma_dens_ie(const orchestration::IntVect& lo, 
                        const orchestration::IntVect& hi,
                        orchestration::FArray4D& U) {
    using namespace orchestration;

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, PRES_VAR_C) =   U(i, j, k, DENS_VAR_C)
                                         * U(i, j, k, EINT_VAR_C)
                                         * eos::GAMMA_M_1_INV;
                U(i, j, k, TEMP_VAR_C) =   U(i, j, k, EINT_VAR_C)
                                         * eos::GGPROD_INV
                                         * eos::SINGLE_SPECIES_A;
            }
        }
    }
}

}

