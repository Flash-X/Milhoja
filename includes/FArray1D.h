/**
 * \file   FArray1D.h
 *
 */

#ifndef FARRAY_1D_H__
#define FARRAY_1D_H__

#include "Grid_REAL.h"

namespace orchestration {

/**
  * \brief  A wrapper class that grants Fortran-like access to a 1D array.
  *
  * A wrapper class around a 1D native C++ data array that allows for
  * Fortran-like access on a 1D array.  Note that the access pattern is also
  * Fortran-style, column-major ordering.
  *
  * \todo Decide if buildScratchArray1D and the associated owner_ private data
  * member is a good design.  I dislike it because the original simple, clean
  * design concept of the FArray1D class was that of a simple wrapper --- it was
  * simply the lens that allowed you to look at the memory in a different way
  * from a distance.  Now, it is or is not the owner of the data that you are
  * looking at depending on how you created it.  This complexity might be
  * reasonable as it offers a simple mechanism to the PUDs and shields them from
  * the data managment.  So far, they only get access to an FArray1D object by
  * calling an infrastructure routine.  These routines manage the underlying
  * resource in such a way that, so far, the PUDs need not think about resource
  * management.
  */
class FArray1D {
public:
    static FArray1D   buildScratchArray1D(const int lo, const int hi);

    FArray1D(Real* data, const int lo);
    ~FArray1D(void);

    FArray1D(FArray1D&)                  = delete;
    FArray1D(const FArray1D&)            = delete;
    FArray1D(FArray1D&&)                 = default;
    FArray1D& operator=(FArray1D&)       = delete;
    FArray1D& operator=(const FArray1D&) = delete;
    FArray1D& operator=(FArray1D&&)      = delete;

    //! Get and set data in a Fortran-style way.
#ifdef ENABLE_OPENACC_OFFLOAD
    #pragma acc routine seq
#endif
    Real& operator()(const int i) const {
        return data_[i-i0_];
    }

#ifdef ENABLE_OPENACC_OFFLOAD
    #pragma acc routine seq
#endif
    Real& at(const int i) const {
        return data_[i-i0_];
    }

    Real* dataPtr(void) {
        return data_;
    }

    const Real* dataPtr(void) const {
        return data_;
    }

private:
    bool         owner_;    //!< Marks if object owns the data.
    Real*        data_;     //!< Pointer to data.
    int          i0_;       //!< Lower bound of first index.
};

}

#endif

