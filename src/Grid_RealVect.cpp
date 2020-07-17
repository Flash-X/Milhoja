#include <iostream>
#include <cmath>

#include "Grid_RealVect.h"
#include "Grid_IntVect.h"

namespace orchestration {

IntVect RealVect::round() const {
    return IntVect(LIST_NDIM(std::round(vect_[0]),std::round(vect_[1]),std::round(vect_[2])));
}
IntVect RealVect::floor() const {
    return IntVect(LIST_NDIM(std::floor(vect_[0]),std::floor(vect_[1]),std::floor(vect_[2])));
}
IntVect RealVect::ceil() const {
    return IntVect(LIST_NDIM(std::ceil(vect_[0]),std::ceil(vect_[1]),std::ceil(vect_[2])));
}

// Scalar multiply a vector (c * V)
RealVect operator* (const Real c, const RealVect& a) {
   return RealVect(LIST_NDIM(a[0]*c, a[1]*c, a[2]*c));
}

// Nice printing of vectors.
std::ostream& operator<< (std::ostream& os, const RealVect& vout) {
    os << '(';
    for (int i=0; i<NDIM; ++i){
        os << vout[i];
        if(i<(NDIM-1)) os << ", ";
    }
    os << ')';
    return os;
}

}
