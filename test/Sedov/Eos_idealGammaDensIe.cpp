#include "Eos.h"

#include "Sedov.h"

namespace Eos {

void idealGammaDensIe(const milhoja::IntVect& lo, 
                      const milhoja::IntVect& hi,
                      milhoja::FArray4D& U) {
    //$milhoja  "U": {
    //$milhoja&    "R": [DENS_VAR, EINT_VAR],
    //$milhoja&    "W": [PRES_VAR, TEMP_VAR]
    //$milhoja& }

    // Taken from FLASH-X Sedov/setup_params file 
    constexpr milhoja::Real   SINGLE_SPECIES_A = 1.0_wp;
    
    // Taken from FLASH-X Physical Constants
    constexpr milhoja::Real   GAS_CONSTANT = 8.3144598e7_wp;  // J/mol/K

    // Derived from runtime parameters/constants
    constexpr milhoja::Real   GGPROD_INV    = (Eos::GAMMA - 1.0_wp) / GAS_CONSTANT;
    constexpr milhoja::Real   GAMMA_M_1_INV = (Eos::GAMMA - 1.0_wp);

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, PRES_VAR) =   U(i, j, k, DENS_VAR)
                                       * U(i, j, k, EINT_VAR)
                                       * GAMMA_M_1_INV;
                U(i, j, k, TEMP_VAR) =   U(i, j, k, EINT_VAR)
                                       * GGPROD_INV
                                       * SINGLE_SPECIES_A;
            }
        }
    }
}

}

