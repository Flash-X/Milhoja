#ifndef GRID_REALVECT_H__
#define GRID_REALVECT_H__

#include "constants.h"
#include "Grid_REAL.h"

namespace orchestration {

class IntVect;

class RealVect
{
  public:
    //generic constructor from 3 Reals
    explicit RealVect (Real x, Real y, Real z) : vect{x,y,z} {}
    //"cast" a RealVect to an IntVect
    explicit operator IntVect() const;

    //get and set values of the internal array with [] operator
    Real& operator[] (int i) { return vect[i]; }
    const Real& operator[] (int i) const { return vect[i]; }

    //check if two vectors are equal element-by-element
    //TODO: change this to a function ??
    bool operator== (const RealVect& b) {
      return AreSame(vect[0],b[0]) && AreSame(vect[1],b[1]) && AreSame(vect[2],b[2]);
    }

    //check if two vectors differ in any place
    bool operator!= (const RealVect& b) {
      return !(*this==b);
    }

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

  private:
    //TODO: >> and << operators

    Real vect[MDIM];
};

// Scalar multiply a vector (c * V), defined in cpp file
RealVect operator* (Real c, const RealVect& a);

} //namespace orchestration



#endif
