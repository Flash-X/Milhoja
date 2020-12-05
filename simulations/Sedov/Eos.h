#ifndef EOS_H__
#define EOS_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "FArray4D.h"

#include "Flash.h"

namespace Eos {
    // lo/hi can be any two corners, including loGC/hiGC
    void idealGamma_dens_ie(const orchestration::IntVect& lo,
                            const orchestration::IntVect& hi,
                            orchestration::FArray4D& solnData);
};

namespace eos {
    // Taken from FLASH-X Sedov/setup_params file 
    constexpr orchestration::Real   SINGLE_SPECIES_A = 1.0_wp;
    
    // Taken from FLASH-X Physical Constants
    constexpr orchestration::Real   GAS_CONSTANT = 8.3144598e7_wp;  // J/mol/K
    
    // Derived from runtime parameters/constants
    constexpr orchestration::Real   GGPROD_INV    = (GAMMA - 1.0_wp) / GAS_CONSTANT;
    constexpr orchestration::Real   GAMMA_M_1_INV = (GAMMA - 1.0_wp);
};

#endif

