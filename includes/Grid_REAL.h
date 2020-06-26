#ifndef GRID_REAL_H__
#define GRID_REAL_H__
#include "Flash.h"

#ifdef REAL_IS_FLOAT
#  undef REAL_IS_FLOAT
#  undef REAL_IS_DOUBLE
#  define REAL_IS_FLOAT 1
/* Use REAL_IS_FLOAT if AMReX was configured to use floats for its real type
   (AMReX was configured with the macro BL_USE_FLOAT).
*/
#else
#  undef REAL_IS_FLOAT
#  undef REAL_IS_DOUBLE
#  define REAL_IS_DOUBLE 1
/* Use REAL_IS_DOUBLE if AMReX was configured to use doubles for its real type
   (AMReX was configured with the macro BL_USE_DOUBLE).
*/
#endif

#ifdef __cplusplus
#include <cfloat>
#else
#include <float.h>
#endif

#ifdef REAL_IS_FLOAT
typedef float grid_real;
// We need to define these to get around a CUDA 9.2 bug that breaks std::numeric_limits
#define GRID_REAL_MIN     FLT_MIN
#define GRID_REAL_MAX     FLT_MAX
#define GRID_REAL_LOWEST -FLT_MAX
#else
typedef double grid_real;
#define GRID_REAL_MIN     DBL_MIN
#define GRID_REAL_MAX     DBL_MAX
#define GRID_REAL_LOWEST -DBL_MAX
#endif

#ifdef __cplusplus
namespace grid {
  using Real = grid_real;
}
inline namespace literals {
  /**
    C++ user literals ``_rt`` for short-hand notations

    Use this to properly add types to constant such as
    ```
    auto const mypi = 3.14_rt;
    auto const sphere_volume = 4_rt / 3_rt * pow(r, 3) * mypi;
    ```
  */
  constexpr grid::Real
  operator"" _rt( long double x )
  {
      return grid::Real( x );
  }

  constexpr grid::Real
  operator"" _rt( unsigned long long int x )
  {
      return grid::Real( x );
  }
}

/* DEV TODO : section for what to do in Fortran */
#endif /* __cplusplus */

#endif
