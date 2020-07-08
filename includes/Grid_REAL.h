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
typedef float orch_real;
// We need to define these to get around a CUDA 9.2 bug that breaks std::numeric_limits
#define GRID_REAL_MIN     FLT_MIN
#define GRID_REAL_MAX     FLT_MAX
#define GRID_REAL_LOWEST -FLT_MAX
#else
typedef double orch_real;
#define GRID_REAL_MIN     DBL_MIN
#define GRID_REAL_MAX     DBL_MAX
#define GRID_REAL_LOWEST -DBL_MAX
#endif

#ifdef __cplusplus
namespace orchestration {
  using Real = orch_real;
}
inline namespace literals {
  /**
    C++ user literals ``_wp`` for short-hand notations

    Use this to properly add types to constant such as
    ```
    Real const mypi = 3.14_wp;
    Real const sphere_volume = 4.0_wp / 3.0_wp * pow(r, 3.0) * mypi;
    ```
  */
  constexpr orchestration::Real
  operator"" _wp( long double x )
  {
      return orchestration::Real( x );
  }

  constexpr orchestration::Real
  operator"" _wp( unsigned long long int x )
  {
      return orchestration::Real( x );
  }
}

/* DEV TODO : section for what to do in Fortran */
#endif /* __cplusplus */

#endif
