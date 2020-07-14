#ifndef GRID_REALVECT_H__
#define GRID_REALVECT_H__

#include "constants.h"
#include "Grid_Macros.h"
#include "Grid_REAL.h"
#include <iosfwd>
#include <stdexcept>

#include <iostream>

namespace orchestration {

class IntVect;

class RealVect
{
  public:

    // Generic constructor, returns a vector with undefined components.
    // TODO: return a default value aka 0?
    explicit RealVect () {}

    // Constructor from NDIM Reals.
    constexpr explicit RealVect (LIST_NDIM(const Real x, const Real y, const Real z)) : vect_{LIST_NDIM(x,y,z)} {}

#if NDIM<3
    // Constructor from 3 Reals.
    explicit RealVect (const Real x, const Real y, const Real z) : vect_{LIST_NDIM(x,y,z)} {
        if(!r_printed_warning) std::cout << "Using deprecated RealVect constructor. Please wrap arguments in LIST_NDIM macro.\n";
        r_printed_warning = true;
    }
#endif

    // Operator to explicitly cast a RealVect to an IntVect.
    // (Implicit cast disabled by `explicit` keyword).
    explicit operator IntVect() const;

    // Get and set values of the internal array with [] operator.
    // Perform bounds check unless GRID_ERRCHECK_OFF is set.
    Real& operator[] (const int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) throw std::logic_error("Index out-of-bounds in RealVect.");
#endif
        return vect_[i];
    }
    const Real& operator[] (const int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) throw std::logic_error("Index out-of-bounds in RealVect.");
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
      return RealVect(LIST_NDIM(vect_[0]+b[0], vect_[1]+b[1], vect_[2]+b[2]));
    }

    // Subtract vectors component-wise.
    RealVect operator- (const RealVect& b) const {
      return RealVect(LIST_NDIM(vect_[0]-b[0], vect_[1]-b[1], vect_[2]-b[2]));
    }

    // Multiply two vectors component-wise.
    RealVect operator* (const RealVect& b) const {
      return RealVect(LIST_NDIM(vect_[0]*b[0], vect_[1]*b[1], vect_[2]*b[2]));
    }

    // Multiply a vector by a scalar (V * c).
    RealVect operator* (const Real c) const {
      return RealVect(LIST_NDIM(vect_[0]*c, vect_[1]*c, vect_[2]*c));
    }

    // Divide two vectors component-wise.
    RealVect operator/ (const RealVect& b) const {
      return RealVect(LIST_NDIM(vect_[0]/b[0], vect_[1]/b[1], vect_[2]/b[2]));
    }

    // Divide a vector by a scalar.
    RealVect operator/ (const Real c) const {
      return RealVect(LIST_NDIM(vect_[0]/c, vect_[1]/c, vect_[2]/c));
    }

    Real product() const {
      return CONCAT_NDIM(vect_[0], * vect_[1], * vect_[2]);
    }

    // Return pointer to underlying array
    const Real* dataPtr() const {
      return vect_;
    }

    friend std::ostream& operator<< (std::ostream& os, const RealVect& vout);

    /* A Note on move/copy sematics.
       */
    RealVect(RealVect&&) = default;
    RealVect& operator=(RealVect&&) = default;
    static bool r_printed_warning;
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
