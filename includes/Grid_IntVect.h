#ifndef GRID_INTVECT_H__
#define GRID_INTVECT_H__

#include "constants.h"
#include "Grid_Macros.h"
#include <vector>
#include <iosfwd>
#include <stdexcept>

#include "Grid_IntTriple.h"

#ifdef GRID_AMREX
#include <AMReX_IntVect.H>
#endif

namespace orchestration {

//forward declaration so we can have a "converter" operator
class RealVect;

class IntVect
{
  public:

    // Generic constructor, returns a vector with undefined components.
    // TODO: return a default value aka 0?
    explicit IntVect () {}

    // Constructor from NDIM ints.
    constexpr explicit IntVect(LIST_NDIM(const int x, const int y, const int z))
        : vect_{LIST_NDIM(x,y,z)} {}

    // Deprecated constructor from int*.
    explicit IntVect (const int* x) : vect_{LIST_NDIM(x[0],x[1],x[2])} {
        throw std::logic_error("IntVect: int* constructor deprecated.");
    }

#if NDIM<3
    // Deprecated constructor from MDIM ints
    explicit IntVect (const int x, const int y, const int z)
        : vect_{LIST_NDIM(x,y,z)} {
        throw std::logic_error("Using deprecated IntVect constructor. Please wrap arguments in LIST_NDIM macro.\n");
    }
#endif

#ifdef GRID_AMREX
    // Constructor from amrex::IntVect
    explicit IntVect (const amrex::IntVect& ain)
        : vect_{LIST_NDIM(ain[0],ain[1],ain[2])} {}

    // Operator to explicitly cast an IntVect to an AMReX IntVect
    explicit operator amrex::IntVect () const {
        return amrex::IntVect(LIST_NDIM(vect_[0],vect_[1],vect_[2]));
    }
#endif

    // Operator to cast an IntVect to a RealVect.
    // (Implicit cast disabled by `explicit` keyword).
    explicit operator RealVect () const;

    // Return as an IntTriple, with 0 in indices over NDIM.
    // Useful for iterating over regions of coordinate space.
    IntTriple asTriple() const {
#if (NDIM==1)
        return IntTriple( vect_[0], 0, 0 );
#elif (NDIM==2)
        return IntTriple( vect_[0], vect_[1], 0 );
#else
        return IntTriple( vect_[0], vect_[1], vect_[2] );
#endif
    }

    // Get and set values of the internal array with [] operator.
    int& operator[] (const int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in IntVect.");
        }
#endif
        return vect_[i];
    }
    const int& operator[] (const int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in IntVect.");
        }
#endif
        return vect_[i];
    }

    // Check if two vectors are equal element-by-element.
    bool operator== (const IntVect& b) const {
      return CONCAT_NDIM(vect_[0]==b[0], && vect_[1]==b[1], && vect_[2]==b[2]);
    }

    // Check if two vectors differ in any place.
    bool operator!= (const IntVect& b) {
      return CONCAT_NDIM(vect_[0]!=b[0], || vect_[1]!=b[1], || vect_[2]!=b[2]);
    }

    //TODO: Potential operators
    // ==, != scalar
    // >, <, etc
    // unary +, -
    // +=, -=, *=, /=
    // + scalar, - scalar
    // min, max, etc

    // Add vectors component-wise.
    IntVect operator+ (const IntVect& b) const {
      return IntVect(LIST_NDIM(vect_[0]+b[0], vect_[1]+b[1], vect_[2]+b[2]));
    }

    // Subtract vectors component-wise.
    IntVect operator- (const IntVect& b) const {
      return IntVect(LIST_NDIM(vect_[0]-b[0], vect_[1]-b[1], vect_[2]-b[2]));
    }

    // Multiply two vectors component-wise.
    IntVect operator* (const IntVect& b) const {
      return IntVect(LIST_NDIM(vect_[0]*b[0], vect_[1]*b[1], vect_[2]*b[2]));
    }

    // Add a scaler to each element.
    IntVect operator+ (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]+c, vect_[1]+c, vect_[2]+c));
    }

    // Subtract a scaler from each element.
    IntVect operator- (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]-c, vect_[1]-c, vect_[2]-c));
    }

    // Multiply a vector by a scalar (V * c).
    IntVect operator* (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]*c, vect_[1]*c, vect_[2]*c));
    }

    // Divide a vector by a scalar.
    IntVect operator/ (const int c) const {
      return IntVect(LIST_NDIM(vect_[0]/c, vect_[1]/c, vect_[2]/c));
    }

    int product() const {
      return CONCAT_NDIM(vect_[0], * vect_[1], * vect_[2]);
    }

    // Return pointer to underlying array
    const int* dataPtr() const {
      return vect_;
    }

    friend std::ostream& operator<< (std::ostream& os, const IntVect& vout);

    /* A Note on move/copy sematics.
       */
    IntVect(IntVect&&) = default;
    IntVect& operator=(IntVect&&) = default;
  private:
    IntVect(IntVect&) = delete;
    IntVect(const IntVect&) = delete;
    IntVect& operator=(IntVect&) = delete;
    IntVect& operator=(const IntVect&) = delete;

    int vect_[NDIM];
};

// Add a scalar to each elements ((c,c,c) + V).
IntVect operator+ (const int c, const IntVect& a);

// Scalar multiply a vector (c * V).
IntVect operator* (const int c, const IntVect& a);



} //namespace orchestration
#endif
