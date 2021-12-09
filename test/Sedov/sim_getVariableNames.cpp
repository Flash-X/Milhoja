#include "Simulation.h"

#include "Sedov.h"
#include "Flash.h"

/**
 * Obtain the names of the physical observables used in the Sedov simulation.
 * The vector is indexed by the variable indices defined in Sedov.h.
 *
 * The Sedov baselines were originally established by comparing against official
 * Flash-X results saved to disk in AMReX's file format.  The variable names in
 * the Flash-X files were given generic names var00XX.  We continue using those
 * names as the baselines have not yet been officially updated to use more
 * specific names.  This is presently sensible as we can still easily compare
 * Sedov results to Flash-X results.
 *
 * \todo  Determine if we should move the variable names to information names.
 *
 * \return A vector of names
 */
std::vector<std::string>   sim::getVariableNames(void) {
    std::vector<std::string>   names{NUNKVAR};

    names[DENS_VAR] = "var0001";
    names[EINT_VAR] = "var0002";
    names[ENER_VAR] = "var0003";
    names[GAMC_VAR] = "var0004";
    names[GAME_VAR] = "var0005";
    names[PRES_VAR] = "var0006";
    names[TEMP_VAR] = "var0007";
    names[VELX_VAR] = "var0008";
    names[VELY_VAR] = "var0009";
    names[VELZ_VAR] = "var0010";

//    names[DENS_VAR] = "dens";
//    names[VELX_VAR] = "velx";
//    names[VELY_VAR] = "vely";
//    names[VELZ_VAR] = "velz";
//    names[PRES_VAR] = "pres";
//    names[ENER_VAR] = "ener";
//    names[GAMC_VAR] = "gamc";
//    names[GAME_VAR] = "game";
//    names[TEMP_VAR] = "temp";
//    names[EINT_VAR] = "eint";

    return names;
}

