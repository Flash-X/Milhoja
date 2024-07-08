/**
 * \file   Milhoja_FArray4D.h
 *
 */

#ifndef MILHOJA_FARRAY_4D_H__
#define MILHOJA_FARRAY_4D_H__

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
  * \brief  A wrapper class that grants Fortran-like access to a 4D array.
  *
  * A wrapper class around a 1D native C++ data array that allows for
  * Fortran-like access on a 4D array.  Note that the access pattern is also
  * Fortran-style, column-major ordering.
  *
  * IMPORTANT: This class should be designed and maintained so that copies of
  * objects of this type can be put into device memory and used such that
  * actions placed on or done by the original objects don't affect the use of
  * the object in the device.
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
  *
  * \todo Add lbound/ubound and size functions that take FArray4D as argument?
  */
class FArray4D {
public:
    static FArray4D   buildScratchArray4D(const IntVect& lo,
                                          const IntVect& hi,
                                          const unsigned int ncomp);

    FArray4D(Real* data, 
             const IntVect& lo, const IntVect& hi,
             const unsigned int ncomp);
    ~FArray4D(void);

    FArray4D(FArray4D&)                  = delete;
    FArray4D(const FArray4D&)            = delete;
    FArray4D(FArray4D&&)                 = default;
    FArray4D& operator=(FArray4D&)       = delete;
    FArray4D& operator=(const FArray4D&) = delete;
    FArray4D& operator=(FArray4D&&)      = delete;

    void reindex(const IntVect& lo);

    /**
     * Get and set data in a Fortran-style way.
     *
     * \todo Should the offline toolchain figure out which routines need to be 
     * decorated this way and write the appropriate directive based on the
     * offloading program model of choice?  Should developers mark this with our
     * own directive so that the toolchain has less work to do?  Since this is 
     * low-level infrastructure, should we just put in the directives protected 
     * by preprocessor so that the compiler just chooses the correct line?
     */
#ifdef ACC_ROUTINE_FOR_METH
    #pragma acc routine seq
#endif
    Real& operator()(const int i, const int j, const int k, const int n) const {
        return data_[  (i-i0_)
                     + (j-j0_)*jstride_
                     + (k-k0_)*kstride_
                     +      n *nstride_];
    }

    /**
     * \todo Is this really necessary?  This is more for people writing kernels
     * so that they have a more pointer-friendly aesthetic? 
     * Couln't the kernel just be written as if the FArray4D object is a direct
     * variable and have the offline toolchain convert it into the simplest,
     * potentially ugly pointer form.
     *
     * f->at(i,j,k,n) better than (*f)(i,j,k,n)?
     */
#ifdef ACC_ROUTINE_FOR_METH
    #pragma acc routine seq
#endif
    Real& at(const int i, const int j, const int k, const int n) const {
        return data_[  (i-i0_)
                     + (j-j0_)*jstride_
                     + (k-k0_)*kstride_
                     +      n *nstride_];
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
    unsigned int nstride_;  //!< Stride for fourth index.
    unsigned int ncomp_;    //!< Size of fourth dimension.
};

}

#endif

