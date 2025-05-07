/**
 * \file   Milhoja_FArray3D.h
 *
 */

#ifndef MILHOJA_FARRAY_3D_H__
#define MILHOJA_FARRAY_3D_H__

#include "Milhoja.h"
#include "Milhoja_real.h"
#include "Milhoja_IntVect.h"

#ifdef MILHOJA_OPENACC_OFFLOADING
#ifndef SUPPRESS_ACC_ROUTINE_FOR_METH_IN_APP
#define ACC_ROUTINE_FOR_METH
#endif
#endif

namespace milhoja {

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

    void reindex(const IntVect& lo);

    /**
     * Get and set data in a Fortran-style way.
     */
#ifdef ACC_ROUTINE_FOR_METH
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

