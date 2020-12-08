#include <string>

#include "Grid_REAL.h"

// Initialize IO's public global variables here
namespace IO {
    //----- INTEGRAL QUANTITIES
    int                     nIntegralQuantities = 0; 
    orchestration::Real*    localIntegralQuantities  = nullptr;
    orchestration::Real*    globalIntegralQuantities = nullptr;
};

// Initialize IO's private global variables here
namespace io {
    //----- INTEGRAL QUANTITIES
    std::string            integralQuantitiesFilename = "DeleteMe.dat";

    orchestration::Real*   blockIntegralQuantities_mass = nullptr;
    orchestration::Real*   blockIntegralQuantities_xmom = nullptr;
    orchestration::Real*   blockIntegralQuantities_ymom = nullptr;
    orchestration::Real*   blockIntegralQuantities_zmom = nullptr;
    orchestration::Real*   blockIntegralQuantities_ener = nullptr;
    orchestration::Real*   blockIntegralQuantities_ke   = nullptr;
    orchestration::Real*   blockIntegralQuantities_eint = nullptr;
    orchestration::Real*   blockIntegralQuantities_magp = nullptr;
};

