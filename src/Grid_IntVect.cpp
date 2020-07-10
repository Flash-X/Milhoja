#include <iostream>

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"

namespace orchestration {
IntVect::operator RealVect() const {
    return RealVect(Real(vect_[0]),Real(vect_[1]),Real(vect_[2]));
}

// Scalar multiply a vector (c * V).
IntVect operator* (const int c, const IntVect& a) {
    return IntVect(a[0]*c, a[1]*c, a[2]*c);
}

// Nice printing of vectors.
std::ostream& operator<< (std::ostream& os, const IntVect& vout) {
    os << '(';
    os << vout[0] << ", " << vout[1] << ", " << vout[2];
    os << ')';
    return os;
}

}
