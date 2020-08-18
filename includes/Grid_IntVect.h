#ifndef GRID_INTVECT_H__
#define GRID_INTVECT_H__

#include "constants.h"
#include "Grid_Macros.h"
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
  * internal array. This operator has bounds checking (unless error checking
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
        : vect_{LIST_NDIM(x,y,z)} {}

    //! Deprecated constructor from int*.
    explicit IntVect (const int* x) : vect_{LIST_NDIM(x[0],x[1],x[2])} {
        throw std::logic_error("IntVect: int* constructor deprecated.");
    }

#if NDIM<3
    //! Deprecated constructor from MDIM ints
    explicit IntVect (const int x, const int y, const int z)
        : vect_{LIST_NDIM(x,y,z)} {
        throw std::logic_error("Using deprecated IntVect constructor. Please wrap arguments in LIST_NDIM macro.\n");
    }
#endif

#ifdef GRID_AMREX
    //! Constructor from amrex::IntVect
    explicit IntVect (const amrex::IntVect& ain)
        : vect_{LIST_NDIM(ain[0],ain[1],ain[2])} {}

    //! Operator to explicitly cast an IntVect to an AMReX IntVect
    explicit operator amrex::IntVect () const {
        return amrex::IntVect(LIST_NDIM(vect_[0],vect_[1],vect_[2]));
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
    int I() const {
        return vect_[0];
    }

    //! Return second element of vector, or 0 if NDIM<2
    int J() const {
#if (NDIM>=2)
        return vect_[1];
#else
        return 0;
#endif
    }

    //! Return third element of vector, or 0 if NDIM<3
    int K() const {
#if (NDIM==3)
        return vect_[2];
#else
        return 0;
#endif
    }

    //! Get and set values of the internal array.
    int& operator[] (const int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in IntVect.");
        }
#endif
        return vect_[i];
    }
    //! Get values of the internal array as const.
    const int& operator[] (const int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in IntVect.");
        }
#endif
        return vect_[i];
    }

    //! Check if two vectors are equal element-by-element.
    bool operator== (const IntVect& b) const {
      return CONCAT_NDIM(vect_[0]==b[0], && vect_[1]==b[1], && vect_[2]==b[2]);
    }

    //! Check if two vectors differ in any place.
    bool operator!= (const IntVect& b) const {
      return CONCAT_NDIM(vect_[0]!=b[0], || vect_[1]!=b[1], || vect_[2]!=b[2]);
    }

    //! Check if two vectors are < in all places.
    bool allLT (const IntVect& b) const {
      return CONCAT_NDIM(vect_[0]<b[0], && vect_[1]<b[1], && vect_[2]<b[2]);
    }

    //! Check if two vectors are <= in all places.
    bool allLE (const IntVect& b) const {
      return CONCAT_NDIM(vect_[0]<=b[0], && vect_[1]<=b[1], && vect_[2]<=b[2]);
    }

    //! Check if two vectors are > in all places.
    bool allGT (const IntVect& b) const {
      return CONCAT_NDIM(vect_[0]>b[0], && vect_[1]>b[1], && vect_[2]>b[2]);
    }

    //! Check if two vectors are >= in all places.
    bool allGE (const IntVect& b) const {
      return CONCAT_NDIM(vect_[0]>=b[0], && vect_[1]>=b[1], && vect_[2]>=b[2]);
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
      return IntVect(LIST_NDIM(vect_[0]+b[0], vect_[1]+b[1], vect_[2]+b[2]));
    }

    //! Subtract vectors component-wise.
    IntVect operator- (const IntVect& b) const {
      return IntVect(LIST_NDIM(vect_[0]-b[0], vect_[1]-b[1], vect_[2]-b[2]));
    }

    //! Multiply two vectors component-wise.
    IntVect operator* (const IntVect& b) const {
      return IntVect(LIST_NDIM(vect_[0]*b[0], vect_[1]*b[1], vect_[2]*b[2]));
    }

    //! Add a scaler to each element.
    IntVect operator+ (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]+c, vect_[1]+c, vect_[2]+c));
    }

    //! Subtract a scaler from each element.
    IntVect operator- (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]-c, vect_[1]-c, vect_[2]-c));
    }

    //! Multiply a vector by a scalar (V * c).
    IntVect operator* (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]*c, vect_[1]*c, vect_[2]*c));
    }

    //! Divide a vector by a scalar.
    IntVect operator/ (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]/c, vect_[1]/c, vect_[2]/c));
    }

    //! Return prduct of components.
    int product() const {
      return CONCAT_NDIM(vect_[0], * vect_[1], * vect_[2]);
    }

    //! Return pointer to underlying array
    const int* dataPtr() const {
      return vect_;
    }

    friend std::ostream& operator<< (std::ostream& os, const IntVect& vout);

  private:

    int vect_[NDIM]; //!< Contains data.
};

//! Add a scalar to each elements ((c,c,c) + V).
IntVect operator+ (const int c, const IntVect& a);

//! Scalar multiply a vector (c * V).
IntVect operator* (const int c, const IntVect& a);



}
#endif
