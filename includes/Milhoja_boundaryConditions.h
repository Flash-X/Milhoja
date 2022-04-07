#ifndef MILHOJA_BOUNDARY_CONDITIONS_H__
#define MILHOJA_BOUNDARY_CONDITIONS_H__

#include "Milhoja.h"

namespace milhoja {

/**
 * Indicate how the boundary conditions at a single domain face should be
 * handled.
 *
 * MILHOJA_PERIODIC - Periodic BC that is handled internally by Milhoja
 * MILHOJA_EXTERNAL_BC - Calling code must provide their own function for
 *                       handling all other types of BCs.
 *
 * It is intended that the corresponding MILHOJA_* integer values in Milhoja.h
 * be used only in Fortran.  Bake the mapping of Fortran onto C++ in the enum
 * class.
 *
 * BC Type   -> MILHOJA_* conversion with
 *              static_cast<int>(BCs)
 * MILHOJA_* -> BC Type conversion with
 *              static_cast<BCs>(MILHOJA_*)
 */
enum class BCs   {Periodic=MILHOJA_PERIODIC,
                  External=MILHOJA_EXTERNAL_BC};
}

#endif

