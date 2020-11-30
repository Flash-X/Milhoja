#ifndef GRID_INTVECT_H__
#define GRID_INTVECT_H__

#include "constants.h"
#include "Grid_Macros.h"
#include "Grid_Axis.h"
#include <vector>
#include <iosfwd>
#include <stdexcept>

#ifdef GRID_AMREX
#include <AMReX_IntVect.H>
#endif

namespace orchestration {

class RealVect;

/**
  * \brief Container for NDIM tuples of integers.
  *
  * There are two methods of indexing in to IntVects. For read-write up
  * to NDIM, use  `operator[]`, which directly obtains reference to the
  * internal data members. This operator has bounds checking (unless error checking
  * is turned off). Alternatively, if MDIM-like behavior is needed, three
  * functions `IntVect::I()`, `IntVect::J()`, and `IntVect::K()` are provided.
  * They return the first, second, or third element of the vector, respectively,
  * or a default value of 0 if trying to get an element above NDIM. These
  * functions should especially be used when writing triple-nested loops that
  * are dimension-agnostic.
  */
class IntVect
{
  public:

    /** \brief Default constructor
      *
      * Returns a vector with undefined components.
      * TODO: return a default value aka 0?
      */
    explicit IntVect () {}

    //! Constructor from NDIM ints.
    constexpr explicit IntVect(LIST_NDIM(const int x, const int y, const int z))
        : LIST_NDIM( i_{x}, j_{y}, k_{z} ) {}

    //! Deprecated constructor from int*.
    explicit IntVect (const int* x)
        : LIST_NDIM( i_{x[0]},j_{x[1]},k_{x[2]}) {
        throw std::logic_error("IntVect: int* constructor deprecated.");
    }

#if NDIM<3
    //! Deprecated constructor from MDIM ints
    explicit IntVect (const int x, const int y, const int z)
        : LIST_NDIM(i_{x},j_{y},k_{z}) {
        throw std::logic_error("Using deprecated IntVect constructor. Please wrap arguments in LIST_NDIM macro.\n");
    }
#endif

#ifdef GRID_AMREX
    //! Constructor from amrex::IntVect
    explicit IntVect (const amrex::IntVect& ain)
        : LIST_NDIM( i_{ain[0]},j_{ain[1]},k_{ain[2]}) {}

    //! Operator to explicitly cast an IntVect to an AMReX IntVect
    explicit operator amrex::IntVect () const {
        return amrex::IntVect(LIST_NDIM(i_,j_,k_));
    }
#endif

    explicit operator RealVect () const;

    // Allow move semantics but no copies.
    IntVect(IntVect&&) = default;
    IntVect& operator=(IntVect&&) = default;
    IntVect(IntVect&) = delete;
    IntVect(const IntVect&) = delete;
    IntVect& operator=(IntVect&) = delete;
    IntVect& operator=(const IntVect&) = delete;

    //! Return first element of vector
#ifdef ENABLE_OPENACC_OFFLOAD
    #pragma acc routine seq
#endif
    int I() const {
        return i_;
    }

    //! Return second element of vector, or 0 if NDIM<2
#ifdef ENABLE_OPENACC_OFFLOAD
    #pragma acc routine seq
#endif
    int J() const {
#if (NDIM>=2)
        return j_;
#else
        return 0;
#endif
    }

    //! Return third element of vector, or 0 if NDIM<3
#ifdef ENABLE_OPENACC_OFFLOAD
    #pragma acc routine seq
#endif
    int K() const {
#if (NDIM==3)
        return k_;
#else
        return 0;
#endif
    }

    //! Get and set values of the internal array.
    int& operator[] (const unsigned int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM) {
            throw std::logic_error("Index out-of-bounds in IntVect.");
        }
#endif
        switch(i) {
            case Axis::I:
                return i_;
#if NDIM>=2
            case Axis::J:
                return j_;
#endif
#if NDIM==3
            case Axis::K:
                return k_;
#endif
        }
        return i_;
    }
    //! Get values of the internal array as const.
    const int& operator[] (const unsigned int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM) {
            throw std::logic_error("Index out-of-bounds in IntVect.");
        }
#endif
        switch(i) {
            case Axis::I:
                return i_;
#if NDIM>=2
            case Axis::J:
                return j_;
#endif
#if NDIM==3
            case Axis::K:
                return k_;
#endif
        }
        return i_;
    }

    //! Check if two vectors are equal element-by-element.
    bool operator== (const IntVect& b) const {
      return CONCAT_NDIM(i_==b.I(), && j_==b.J(), && k_==b.K());
    }

    //! Check if two vectors differ in any place.
    bool operator!= (const IntVect& b) const {
      return CONCAT_NDIM(i_!=b.I(), || j_!=b.J(), || k_!=b.K());
    }

    //TODO: Potential operators
    // ==, != scalar
    // >, <, etc
    // unary +, -
    // +=, -=, *=, /=
    // + scalar, - scalar
    // min, max, etc

    //! Add vectors component-wise.
    IntVect operator+ (const IntVect& b) const {
      return IntVect(LIST_NDIM(i_+b.I(), j_+b.J(), k_+b.K()));
    }

    //! Subtract vectors component-wise.
    IntVect operator- (const IntVect& b) const {
      return IntVect(LIST_NDIM(i_-b.I(), j_-b.J(), k_-b.K()));
    }

    //! Multiply two vectors component-wise.
    IntVect operator* (const IntVect& b) const {
      return IntVect(LIST_NDIM(i_*b.I(), j_*b.J(), k_*b.K()));
    }

    //! Add a scaler to each element.
    IntVect operator+ (const int c) const {
      return IntVect(LIST_NDIM(i_+c, j_+c, k_+c));
    }

    //! Subtract a scaler from each element.
    IntVect operator- (const int c) const {
      return IntVect(LIST_NDIM(i_-c, j_-c, k_-c));
    }

    //! Multiply a vector by a scalar (V * c).
    IntVect operator* (const int c) const {
      return IntVect(LIST_NDIM(i_*c, j_*c, k_*c));
    }

    //! Divide a vector by a scalar.
    IntVect operator/ (const int c) const {
      return IntVect(LIST_NDIM(i_/c, j_/c, k_/c));
    }

    //! Return prduct of components.
    int product() const {
      return CONCAT_NDIM(i_, * j_, * k_);
    }

    //! \todo change how IntVect can return as a true vector
    //const int* dataPtr() const {
    //  return vect_;
    //}

    friend std::ostream& operator<< (std::ostream& os, const IntVect& vout);

  private:

    int LIST_NDIM(i_,j_,k_); //!< Contains data.
};

//! Add a scalar to each elements ((c,c,c) + V).
IntVect operator+ (const int c, const IntVect& a);

//! Scalar multiply a vector (c * V).
IntVect operator* (const int c, const IntVect& a);



}
#endif
