#include "Milhoja_RealVect.h"

#include <iostream>
#include <cmath>

#include "Milhoja_IntVect.h"

namespace milhoja {

//! Convert to IntVect via rounding each element
IntVect RealVect::round() const {
    return IntVect(LIST_NDIM(std::round(i_),
                             std::round(j_),
                             std::round(k_)));
}
//! Convert to IntVect via floor
IntVect RealVect::floor() const {
    return IntVect(LIST_NDIM(std::floor(i_),
                             std::floor(j_),
                             std::floor(k_)));
}
//! Convert to IntVect via ceil
IntVect RealVect::ceil() const {
    return IntVect(LIST_NDIM(std::ceil(i_),
                             std::ceil(j_),
                             std::ceil(k_)));
}

//! Scalar multiply a vector (c * V)
RealVect operator* (const Real c, const RealVect& a) {
   return RealVect(LIST_NDIM(a.I()*c, a.J()*c, a.K()*c));
}

//! Nice printing of vectors.
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
