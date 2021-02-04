/**
 * \file   FArray3D.h
 *
 */

#ifndef FARRAY_3D_H__
#define FARRAY_3D_H__

#include "Grid_REAL.h"
#include "Grid_IntVect.h"

namespace orchestration {

/**
  * \brief  A wrapper class that grants Fortran-like access to a 3D array.
  *
  * A wrapper class around a 1D native C++ data array that allows for
  * Fortran-like access on a 3D array.  Note that the access pattern is also
  * Fortran-style, column-major ordering.
  *
  * IMPORTANT: This class should be designed and maintained so that copies of
  * objects of this type can be put into device memory and used such that
  * actions placed on or done by the original objects don't affect the use of
  * the object in the device.
  */
class FArray3D {
public:
    static FArray3D   buildScratchArray(const IntVect& lo, const IntVect& hi);

    FArray3D(Real* data, const IntVect& lo, const IntVect& hi);
    ~FArray3D(void);

    FArray3D(FArray3D&)                  = delete;
    FArray3D(const FArray3D&)            = delete;
    FArray3D(FArray3D&&)                 = default;
    FArray3D& operator=(FArray3D&)       = delete;
    FArray3D& operator=(const FArray3D&) = delete;
    FArray3D& operator=(FArray3D&&)      = delete;

    /**
     * Get and set data in a Fortran-style way.
     */
#ifdef ENABLE_OPENACC_OFFLOAD
    #pragma acc routine seq
#endif
    Real& operator()(const int i, const int j, const int k) const {
        return data_[  (i-i0_)
                     + (j-j0_)*jstride_
                     + (k-k0_)*kstride_];
    }

private:
    // So that objects of this class can be copied directly into device memory
    // easily, the data members in this class must be mananged and maintained
    // so that we need not transfer any data other than the data pointed to by
    // data_ to device memory.
    bool         owner_;    //!< Marks if object owns the data.
    Real*        data_;     //!< Pointer to data.
    int          i0_;       //!< Lower bound of first index.
    int          j0_;       //!< Lower bound of second index.
    int          k0_;       //!< Lower bound of third index.
    unsigned int jstride_;  //!< Stride for second index.
    unsigned int kstride_;  //!< Stride for third index.
};

}

#endif

