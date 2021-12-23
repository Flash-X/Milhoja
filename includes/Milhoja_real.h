#ifndef MILHOJA_REAL_H__
#define MILHOJA_REAL_H__

#include "Milhoja.h"

#ifdef MILHOJA_REAL_IS_FLOAT
#  ifdef MILHOJA_REAL_IS_DOUBLE
#    error Cannot define Real as both Double and Float.
#  else
#    undef MILHOJA_REAL_IS_FLOAT
#    undef MILHOJA_REAL_IS_DOUBLE
#    define MILHOJA_REAL_IS_FLOAT 1
/*   Use MILHOJA_REAL_IS_FLOAT if AMReX was configured to use floats for its real type
     (AMReX was configured with the macro BL_USE_FLOAT).
*/
#  endif
#else
#  ifdef MILHOJA_REAL_IS_DOUBLE
#    undef MILHOJA_REAL_IS_FLOAT
#    undef MILHOJA_REAL_IS_DOUBLE
#    define MILHOJA_REAL_IS_DOUBLE 1
/* Use MILHOJA_REAL_IS_DOUBLE if AMReX was configured to use doubles for its real type
   (AMReX was configured with the macro BL_USE_DOUBLE).
*/
#  else
#    error Please define either MILHOJA_REAL_IS_DOUBLE or MILHOJA_REAL_IS_FLOAT to match the configuration of AMReX.
#  endif
#endif

#ifdef __cplusplus
#include <cfloat>
#else
#include <float.h>
#endif

#ifdef MILHOJA_REAL_IS_FLOAT
typedef float orch_real;
#else
typedef double orch_real;
#endif

#ifdef __cplusplus
namespace milhoja {
  using Real = orch_real;
}
inline namespace literals {
  /**
    Use _wp to properly add types to literals. For example:
    ```
    const Real mypi = 3.14_wp;
    Real sphere_volume = 4.0_wp / 3.0_wp * Real(pow(r, 3.0)) * mypi;
    ```
  */
  constexpr milhoja::Real
  operator"" _wp( long double x )
  {
      return milhoja::Real( x );
  }

  constexpr milhoja::Real
  operator"" _wp( unsigned long long int x )
  {
      return milhoja::Real( x );
  }
}

/* DEV TODO : section for what to do in Fortran */
#endif

#endif
