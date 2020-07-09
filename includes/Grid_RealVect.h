#ifndef GRID_REALVECT_H__
#define GRID_REALVECT_H__

#include "constants.h"
#include "Grid_REAL.h"

namespace orchestration {

class IntVect;

class RealVect
{
  public:
    //generic constructor
    explicit RealVect () {}
    //constructor from 3 Reals
    explicit RealVect (const Real x, const Real y, const Real z) : vect{x,y,z} {}
    //"cast" a RealVect to an IntVect
    explicit operator IntVect() const;

    //get and set values of the internal array with [] operator
    Real& operator[] (const int i) { return vect[i]; }
    const Real& operator[] (const int i) const { return vect[i]; }

    //TODO: Potential operators
    // ==, != scalar
    // >, <, etc
    // unary +, -
    // +=, -=, *=, /=
    // + scalar, - scalar
    // min, max, etc

    //add two vectors
    RealVect operator+ (const RealVect& b) const {
      return RealVect(vect[0]+b[0], vect[1]+b[1], vect[2]+b[2]);
    }

    //subtract one vector from another
    RealVect operator- (const RealVect& b) const {
      return RealVect(vect[0]-b[0], vect[1]-b[1], vect[2]-b[2]);
    }

    //multiply two vectors component-wise
    RealVect operator* (const RealVect& b) const {
      return RealVect(vect[0]*b[0], vect[1]*b[1], vect[2]*b[2]);
    }

    //multiply a vector by a scalar (V * c)
    RealVect operator* (const Real c) const {
      return RealVect(vect[0]*c, vect[1]*c, vect[2]*c);
    }

    //divide a vector by a scalar
    RealVect operator/ (const Real c) const {
      return RealVect(vect[0]/c, vect[1]/c, vect[2]/c);
    }

    //move constructors
    RealVect(RealVect&&) = default;
    RealVect& operator=(RealVect&&) = default;
  private:
    //copy constructors disabled for now
    RealVect(RealVect&) = delete;
    RealVect(const RealVect&) = delete;
    RealVect& operator=(RealVect&) = delete;
    RealVect& operator=(const RealVect&) = delete;

    //TODO: >> and << operators

    Real vect[MDIM];
};

// Scalar multiply a vector (c * V), defined in cpp file
RealVect operator* (const Real c, const RealVect& a);

} //namespace orchestration



#endif
