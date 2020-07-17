#include <iostream>
#include <iomanip>
#include <sstream>

#include "Grid_RealVect.h"
#include "Grid_IntVect.h"

namespace orchestration {

RealVect::operator IntVect() const {
    return IntVect(LIST_NDIM(int(vect_[0]),int(vect_[1]),int(vect_[2])));
}

// Scalar multiply a vector (c * V)
RealVect operator* (const Real c, const RealVect& a) {
   return RealVect(LIST_NDIM(a[0]*c, a[1]*c, a[2]*c));
}

// Nice printing of vectors.
std::ostream& operator<< (std::ostream& os, const RealVect& vout) {
    os << '(';
    for (int i=0; i<NDIM; ++i){
        std::ostringstream elem;
        elem << std::fixed;
        elem << std::setprecision(4);
        elem << vout[i];
        os << elem.str();
        if(i<(NDIM-1)) os << ", ";
    }
    os << ')';
    return os;
}

}
