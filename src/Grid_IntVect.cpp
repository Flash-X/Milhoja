#include <iostream>

#include "Grid_IntVect.h"
#include "Grid_RealVect.h"

namespace orchestration {
IntVect::operator RealVect() const {
    return RealVect(LIST_NDIM(Real(vect_[0]),Real(vect_[1]),Real(vect_[2])));
}

// Add a scalar ((c,c,c) + V).
IntVect operator+ (const int c, const IntVect& a) {
    return IntVect(LIST_NDIM(a[0]+c, a[1]+c, a[2]+c));
}

// Scalar multiply a vector (c * V).
IntVect operator* (const int c, const IntVect& a) {
    return IntVect(LIST_NDIM(a[0]*c, a[1]*c, a[2]*c));
}

// Nice printing of vectors.
std::ostream& operator<< (std::ostream& os, const IntVect& vout) {
    os << '(';
    for(int i=0; i<NDIM; ++i) {
        os << vout[i];
        if(i<(NDIM-1)) os << ", ";
    }
    os << ')';
    return os;
}

bool IntVect::i_printed_warning = false;

}
