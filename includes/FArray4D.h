/**
 * \file   FArray4D.h
 *
 * \brief  A wrapper class that grants Fortran-like access to a 4D array.
 *
 * A wrapper class around a 1D native C++ data array that allows for
 * Fortran-like access on a 4D array.  Note that the access pattern is also
 * Fortran-style, column-major ordering.
 *
 * \todo Decide if buildScratchArray4D and the associated owner_ private data
 * member is a good design.  I dislike it because the original simple, clean
 * design concept of the FArray4D class was that of a simple wrapper --- it was
 * simply the lens that allowed you to look at the memory in a different way
 * from a distance.  Now, it is or is not the owner of the data that you are
 * looking at depending on how you created it.  This complexity might be
 * reasonable as it offers a simple mechanism to the PUDs and shields them from
 * the data managment.  So far, they only get access to an FArray4D object by
 * calling an infrastructure routine.  These routines manage the underlying
 * resource in such a way that, so far, the PUDs need not think about resource
 * management.
 */

#ifndef FARRAY_4D_H__
#define FARRAY_4D_H__

#include "Grid_REAL.h"
#include "Grid_Axis.h"
#include "Grid_IntTriple.h"

namespace orchestration {

class FArray4D {
public:
    static FArray4D   buildScratchArray4D(const IntTriple& lo,
                                          const IntTriple& hi,
                                          const unsigned int ncomp);

    FArray4D(Real* data, 
             const IntTriple& lo, const IntTriple& hi,
             const unsigned int ncomp);
    ~FArray4D(void);

    FArray4D(FArray4D&)                  = delete;
    FArray4D(const FArray4D&)            = delete;
    FArray4D(FArray4D&&)                 = default;
    FArray4D& operator=(FArray4D&)       = delete;
    FArray4D& operator=(const FArray4D&) = delete;
    FArray4D& operator=(FArray4D&&)      = delete;

    Real& operator()(const int i, const int j, const int k, const int n) const {
        return data_[  (i-i0_)
                     + (j-j0_)*jstride_
                     + (k-k0_)*kstride_
                     +      n *nstride_];
    }

private:
    bool         owner_;
    Real*        data_;
    int          i0_;
    int          j0_;
    int          k0_;
    unsigned int jstride_;
    unsigned int kstride_;
    unsigned int nstride_;
    unsigned int ncomp_;
};

}

#endif

