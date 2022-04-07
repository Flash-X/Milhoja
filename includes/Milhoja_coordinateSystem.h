#ifndef MILHOJA_COORDINATE_SYSTEM_H__
#define MILHOJA_COORDINATE_SYSTEM_H__

#include "Milhoja.h"

namespace milhoja {

/**
 * The coordinate system used to define the domain.
 *
 * It is intended that the MILHOJA_* integer values in Milhoja.h be used only in
 * Fortran.  Bake the mapping of Fortran onto C++ in the enum class.
 *
 * CoordSys  -> MILHOJA_* conversion with
 *              static_cast<int>(myCoordSys)
 * MILHOJA_* -> CoordSys conversion with
 *              static_cast<CoordSys>(MILHOJA_*)
 */
enum class CoordSys {Cartesian=MILHOJA_CARTESIAN,
                     Cylindrical=MILHOJA_CYLINDRICAL,
                     Spherical=MILHOJA_SPHERICAL};

}

#endif

