#ifndef MILHOJA_INTERPOLATOR_H__
#define MILHOJA_INTERPOLATOR_H__

#include "Milhoja.h"

namespace milhoja {

/**
 * The union of all interpolators available across all grid backends.  This
 * implies that an interpolator might be available for more than one backend,
 * but might not be available for all backends.  That an interpolator is
 * available for more than one backend does not imply that the interpolator
 * functions identically across backends.  Rather, users should refer to the
 * documentation of the AMR implementation associated with a target grid backend
 * for detailed information.
 *
 * AMReX interpolators
 * -------------------
 * Interpolator                 AMReX Class Name
 * -----------------------------------------------
 * CellConservativeLinear       CellConservativeLinear
 * CellConservativeProtected    CellConservativeProtected
 * CellConservativeQuartic      CellConservativeQuartic
 * CellPiecewiseConstant        PCInterp
 * CellBilinear                 CellBilinear
 * CellQuadratic                CellQuadratic
 * NodeBilinear                 NodeBilinear
 * FaceLinear                   FaceLinear
 * FaceDivergenceFree           FaceDivFree
 *
 * It is intended that the corresponding MILHOJA_* integer values in Milhoja.h
 * be used only in Fortran.
 *
 * Intepolator -> MILHOJA_* conversion with
 *         static_cast<int>(Interpolator)
 * MILHOJA_* -> BC Type conversion with
 *         static_cast<Interpolator>(MILHOJA_*)
 */
// Bake the mapping of Fortran onto C++ in the enum class
enum class Interpolator   {CellConservativeLinear=MILHOJA_CELL_CONSERVATIVE_LINEAR,
                           CellConservativeProtected=MILHOJA_CELL_CONSERVATIVE_PROTECTED,
                           CellConservativeQuartic=MILHOJA_CELL_CONSERVATIVE_QUARTIC,
                           CellPiecewiseConstant=MILHOJA_CELL_PIECEWISE_CONSTANT,
                           CellBilinear=MILHOJA_CELL_BILINEAR,
                           CellQuadratic=MILHOJA_CELL_QUADRATIC,
                           NodeBilinear=MILHOJA_NODE_BILINEAR,
                           FaceLinear=MILHOJA_FACE_LINEAR,
                           FaceDivergenceFree=MILHOJA_FACE_DIVERGENCE_FREE};
}

#endif

