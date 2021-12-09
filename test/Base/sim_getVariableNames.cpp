#include "Simulation.h"

#include "Base.h"
#include "Flash.h"

/**
 * Obtain the names of the physical observables used in the Base test case.
 *
 * \return A vector of names indexed by the variable indices defined in Base.h
 */
std::vector<std::string>   sim::getVariableNames(void) {
    std::vector<std::string>   names{NUNKVAR};

    names[DENS_VAR] = "dens";
    names[ENER_VAR] = "ener";

    return names;
}

