/**
 * \file   FArray2D.h
 *
 */

#ifndef FARRAY_2D_H__
#define FARRAY_2D_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"

namespace orchestration {

/**
  * \brief  A wrapper class that grants Fortran-like access to a 2D array.
  *
  * A wrapper class around a 1D native C++ data array that allows for
  * Fortran-like access on a 2D array.  Note that the access pattern is also
  * Fortran-style, column-major ordering.
  */
class FArray2D {
public:
    FArray2D(Real* data, const IntVect& lo, const IntVect& hi);
    ~FArray2D(void);

    FArray2D(FArray2D&)                  = delete;
    FArray2D(const FArray2D&)            = delete;
    FArray2D(FArray2D&&)                 = default;
    FArray2D& operator=(FArray2D&)       = delete;
    FArray2D& operator=(const FArray2D&) = delete;
    FArray2D& operator=(FArray2D&&)      = delete;

    //! Get and set data in a Fortran-style way.
    Real& operator()(const int i, const int j) const {
        return data_[  (i-i0_)
                     + (j-j0_)*jstride_];
    }

private:
    Real*        data_;     //!< Pointer to data.
    int          i0_;       //!< Lower bound of first index.
    int          j0_;       //!< Lower bound of second index.
    unsigned int jstride_;  //!< Stride for second index.
};

}

#endif

