#ifndef GRID_INTVECT_H__
#define GRID_INTVECT_H__

#include "constants.h"

namespace orchestration {

//forward declaration so we can have a "converter" operator
class RealVect;

class IntVect
{
  public:
    //genric constructor
    explicit IntVect () {}
    //constructor from 3 ints
    explicit IntVect (const int x, const int y, const int z) : vect{x,y,z} {}
    //"cast" an IntVect to a RealVect
    explicit operator RealVect () const;

    //get and set values of the internal array with [] operator
    int& operator[] (const int i) { return vect[i]; }
    const int& operator[] (const int i) const { return vect[i]; }

    //check if two vectors are equal element-by-element
    bool operator== (const IntVect& b) {
      return vect[0]==b[0] && vect[1]==b[1] && vect[2]==b[2];
    }

    //check if two vectors differ in any place
    bool operator!= (const IntVect& b) {
      return vect[0]!=b[0] || vect[1]!=b[1] || vect[2]!=b[2];
    }

    //TODO: Potential operators
    // ==, != scalar
    // >, <, etc
    // unary +, -
    // +=, -=, *=, /=
    // + scalar, - scalar
    // min, max, etc

    //add two vectors
    IntVect operator+ (const IntVect& b) const {
      return IntVect(vect[0]+b[0], vect[1]+b[1], vect[2]+b[2]);
    }

    //subtract one vector from another
    IntVect operator- (const IntVect& b) const {
      return IntVect(vect[0]-b[0], vect[1]-b[1], vect[2]-b[2]);
    }

    //multiply two vectors component-wise
    IntVect operator* (const IntVect& b) const {
      return IntVect(vect[0]*b[0], vect[1]*b[1], vect[2]*b[2]);
    }

    //multiply a vector by a scalar (V * c)
    IntVect operator* (const int c) const {
      return IntVect(vect[0]*c, vect[1]*c, vect[2]*c);
    }

    //divide a vector by a scalar
    IntVect operator/ (const int c) const {
      return IntVect(vect[0]/c, vect[1]/c, vect[2]/c);
    }

    //move and copy constructors all should be fine
    IntVect(IntVect&&) = default;
    IntVect& operator=(IntVect&&) = default;
    IntVect(IntVect&) = default;
    IntVect(const IntVect&) = default;
    IntVect& operator=(IntVect&) = default;
    IntVect& operator=(const IntVect&) = default;
  private:
    //TODO: >> and << operators

    int vect[MDIM];
};

// Scalar multiply a vector (c * V), defined in cpp file
IntVect operator* (const int c, const IntVect& a);

} //namespace orchestration



#endif
