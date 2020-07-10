#ifndef GRID_REALVECT_H__
#define GRID_REALVECT_H__

#include "constants.h"
#include "Grid_REAL.h"
#include <iosfwd>
#include <stdexcept>

namespace orchestration {

class IntVect;

class RealVect
{
  public:

    // Generic constructor, returns a vector with undefined components.
    // TODO: return a default value aka 0?
    explicit RealVect () {}

    // Constructor from 3 Reals.
    constexpr explicit RealVect (const Real x, const Real y, const Real z) : vect_{x,y,z} {}

    // Operator to explicitly cast a RealVect to an IntVect.
    // (Implicit cast disabled by `explicit` keyword).
    explicit operator IntVect() const;

    // Get and set values of the internal array with [] operator.
    // Perform bounds check unless GRID_ERRCHECK_OFF is set.
    Real& operator[] (const int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=MDIM || i<0) throw std::logic_error("Index out-of-bounds in RealVect.");
#endif
        return vect_[i];
    }
    const Real& operator[] (const int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=MDIM || i<0) throw std::logic_error("Index out-of-bounds in RealVect.");
#endif
        return vect_[i];
    }

    //TODO: Potential operators
    // ==, != scalar
    // >, <, etc
    // unary +, -
    // +=, -=, *=, /=
    // + scalar, - scalar
    // min, max, etc

    // Add vectors component-wise.
    RealVect operator+ (const RealVect& b) const {
      return RealVect(vect_[0]+b[0], vect_[1]+b[1], vect_[2]+b[2]);
    }

    // Subtract vectors component-wise.
    RealVect operator- (const RealVect& b) const {
      return RealVect(vect_[0]-b[0], vect_[1]-b[1], vect_[2]-b[2]);
    }

    // Multiply two vectors component-wise.
    RealVect operator* (const RealVect& b) const {
      return RealVect(vect_[0]*b[0], vect_[1]*b[1], vect_[2]*b[2]);
    }

    // Multiply a vector by a scalar (V * c).
    RealVect operator* (const Real c) const {
      return RealVect(vect_[0]*c, vect_[1]*c, vect_[2]*c);
    }

    // Divide a vector by a scalar.
    RealVect operator/ (const Real c) const {
      return RealVect(vect_[0]/c, vect_[1]/c, vect_[2]/c);
    }

    friend std::ostream& operator<< (std::ostream& os, const RealVect& vout);

    /* A Note on move/copy sematics.
       */
    RealVect(RealVect&&) = default;
    RealVect& operator=(RealVect&&) = default;
  private:
    RealVect(RealVect&) = delete;
    RealVect(const RealVect&) = delete;
    RealVect& operator=(RealVect&) = delete;
    RealVect& operator=(const RealVect&) = delete;

    //TODO: >> and << operators

    Real vect_[MDIM];
};

// Scalar multiply a vector (c * V).
RealVect operator* (const Real c, const RealVect& a);



} //namespace orchestration
#endif
