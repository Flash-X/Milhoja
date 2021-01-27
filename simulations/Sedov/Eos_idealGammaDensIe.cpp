#include "Eos.h"

#include "Flash.h"
#include "Flash_par.h"

namespace Eos {

void idealGammaDensIe(const orchestration::IntVect& lo, 
                      const orchestration::IntVect& hi,
                      orchestration::FArray4D& U) {
    // Taken from FLASH-X Sedov/setup_params file 
    constexpr orchestration::Real   SINGLE_SPECIES_A = 1.0_wp;
    
    // Taken from FLASH-X Physical Constants
    constexpr orchestration::Real   GAS_CONSTANT = 8.3144598e7_wp;  // J/mol/K
    
    // Derived from runtime parameters/constants
    constexpr orchestration::Real   GGPROD_INV    = (rp_Eos::GAMMA - 1.0_wp) / GAS_CONSTANT;
    constexpr orchestration::Real   GAMMA_M_1_INV = (rp_Eos::GAMMA - 1.0_wp);

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, PRES_VAR_C) =   U(i, j, k, DENS_VAR_C)
                                         * U(i, j, k, EINT_VAR_C)
                                         * GAMMA_M_1_INV;
                U(i, j, k, TEMP_VAR_C) =   U(i, j, k, EINT_VAR_C)
                                         * GGPROD_INV
                                         * SINGLE_SPECIES_A;
            }
        }
    }
}

}

