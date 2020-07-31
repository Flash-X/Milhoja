/**
 * \file   FArray4D.h
 *
 * \brief  A wrapper class that grants Fortran-like access to a 4D array.
 *
 * A wrapper class around a 1D native C++ data array that allows for
 * Fortran-like access on a 4D array.  Note that the access pattern is also
 * Fortran-style row major.
 *
 * As presently implemented, this wrapper is given a pointer to the data, which
 * is  structured with a 1D contiguous layout in memory.  However, the object
 * that is instantiated does not assume ownership of the memory resource in any
 * way.  Implicit in this is that such objects should only exist so long as
 * access to and correctness of the underlying data is guaranteed.
 */

#ifndef FARRAY_4D_H__
#define FARRAY_4D_H__

#include "Grid_REAL.h"
#include "Grid_Axis.h"
#include "Grid_IntVect.h"

namespace orchestration {

// TODO Update to 3D once Grid allows for coding up
//      3D loops.
// TODO: Add in a second constructor that does not take a pointer, but rather
// dynamically allocates the needed array and deallocates during destruction of
// the wrapper object?
class FArray4D {
public:
    static FArray4D   buildScratchArray4D(const IntVect& begin, const IntVect& end,
                                          const unsigned int ncomp);

    FArray4D(Real* data, 
             const IntVect& begin, const IntVect& end,
             const unsigned int ncomp);
    ~FArray4D(void);

    FArray4D(FArray4D&)                  = delete;
    FArray4D(const FArray4D&)            = delete;
    FArray4D(FArray4D&&)                 = default;
    FArray4D& operator=(FArray4D&)       = delete;
    FArray4D& operator=(const FArray4D&) = delete;
    FArray4D& operator=(FArray4D&&)      = delete;

//    Real& operator()(int i, int j, int k, int n) const {
    Real& operator()(int i, int j, int n) const {
        return data_[(i-i0_) + (j-j0_)*jstride_ + n*nstride_];
//        return data_[(i-i0_) + (j-j0_)*jstride_ + (k-k0_)*kstride_ + n*nstride_];
    }

private:
    bool         owner_;
    Real*        data_;
    int          i0_;
    int          j0_;
//    int          k0_;
    unsigned int jstride_;
//    unsigned int kstride_;
    unsigned int nstride_;
    unsigned int ncomp_;
};

}

#endif

