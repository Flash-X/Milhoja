#ifndef GRID_INTVECT_H__
#define GRID_INTVECT_H__

#include "constants.h"
#include <iosfwd>
#include <stdexcept>

namespace orchestration {

//forward declaration so we can have a "converter" operator
class RealVect;

class IntVect
{
  public:

    // Generic constructor, returns a vector with undefined components.
    // TODO: return a default value aka 0?
    explicit IntVect () {}

    // Constructor from 3 ints
    constexpr explicit IntVect (const int x, const int y, const int z) : vect_{x,y,z} {}

    // Operator to explicitly cast an IntVect to a RealVect.
    // (Implicit cast disabled by `explicit` keyword).
    explicit operator RealVect () const;

    // Get and set values of the internal array with [] operator.
    int& operator[] (const int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=MDIM || i<0) throw std::logic_error("Index out-of-bounds in IntVect.");
#endif
        return vect_[i];
    }
    const int& operator[] (const int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=MDIM || i<0) throw std::logic_error("Index out-of-bounds in IntVect.");
#endif
        return vect_[i];
    }

    // Check if two vectors are equal element-by-element.
    bool operator== (const IntVect& b) {
      return vect_[0]==b[0] && vect_[1]==b[1] && vect_[2]==b[2];
    }

    // Check if two vectors differ in any place.
    bool operator!= (const IntVect& b) {
      return vect_[0]!=b[0] || vect_[1]!=b[1] || vect_[2]!=b[2];
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
      return IntVect(vect_[0]+b[0], vect_[1]+b[1], vect_[2]+b[2]);
    }

    // Subtract vectors component-wise.
    IntVect operator- (const IntVect& b) const {
      return IntVect(vect_[0]-b[0], vect_[1]-b[1], vect_[2]-b[2]);
    }

    // Multiply two vectors component-wise.
    IntVect operator* (const IntVect& b) const {
      return IntVect(vect_[0]*b[0], vect_[1]*b[1], vect_[2]*b[2]);
    }

    // Multiply a vector by a scalar (V * c).
    IntVect operator* (const int c) const {
      return IntVect(vect_[0]*c, vect_[1]*c, vect_[2]*c);
    }

    // Divide a vector by a scalar.
    IntVect operator/ (const int c) const {
      return IntVect(vect_[0]/c, vect_[1]/c, vect_[2]/c);
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

    int vect_[MDIM];
};

// Scalar multiply a vector (c * V).
IntVect operator* (const int c, const IntVect& a);



} //namespace orchestration
#endif
