#include "Milhoja_IntVect.h"

#include <iostream>

#include "Milhoja_RealVect.h"

namespace milhoja {

//! Cast an IntVect to RealVect
IntVect::operator RealVect() const {
    return RealVect(LIST_NDIM(Real(i_),Real(j_),Real(k_)));
}

//! Add a scalar ((c,c,c) + V).
IntVect operator+ (const int c, const IntVect& a) {
    return IntVect(LIST_NDIM(a.I()+c, a.J()+c, a.K()+c));
}

//! Scalar multiply a vector (c * V).
IntVect operator* (const int c, const IntVect& a) {
    return IntVect(LIST_NDIM(a.I()*c, a.J()*c, a.K()*c));
}

//! Nice printing of vectors.
std::ostream& operator<< (std::ostream& os, const IntVect& vout) {
    os << '(';
    for(int i=0; i<NDIM; ++i) {
        os << vout[i];
        if(i<(NDIM-1)) os << ", ";
    }
    os << ')';
    return os;
}

}
