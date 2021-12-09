#ifndef ENABLE_OPENACC_OFFLOAD
#error "This file should only be compiled if using OpenACC offloading"
#endif

#include "Eos.h"

#include "Flash.h"
#include "Flash_par.h"

namespace Eos {

void idealGammaDensIe_oacc_summit(const orchestration::IntVect* lo_d, 
                                  const orchestration::IntVect* hi_d,
                                  orchestration::FArray4D* U_d) {
    // Taken from FLASH-X Sedov/setup_params file 
    constexpr orchestration::Real   SINGLE_SPECIES_A = 1.0_wp;
    
    // Taken from FLASH-X Physical Constants
    constexpr orchestration::Real   GAS_CONSTANT = 8.3144598e7_wp;  // J/mol/K
    
    // Derived from runtime parameters/constants
    constexpr orchestration::Real   GGPROD_INV    = (rp_Eos::GAMMA - 1.0_wp) / GAS_CONSTANT;
    constexpr orchestration::Real   GAMMA_M_1_INV = (rp_Eos::GAMMA - 1.0_wp);

    int     i_s = lo_d->I();
    int     j_s = lo_d->J();
    int     k_s = lo_d->K();

    int     i_e = hi_d->I();
    int     j_e = hi_d->J();
    int     k_e = hi_d->K();

    #pragma acc loop vector collapse(3)
    for         (int k=k_s; k<=k_e; ++k) {
        for     (int j=j_s; j<=j_e; ++j) {
            for (int i=i_s; i<=i_e; ++i) {
                U_d->at(i, j, k, PRES_VAR) =   U_d->at(i, j, k, DENS_VAR)
                                             * U_d->at(i, j, k, EINT_VAR)
                                             * GAMMA_M_1_INV;
                U_d->at(i, j, k, TEMP_VAR) =   U_d->at(i, j, k, EINT_VAR)
                                             * GGPROD_INV
                                             * SINGLE_SPECIES_A;
            }
        }
    }
}

}

