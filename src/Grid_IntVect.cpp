#include "Grid_IntVect.h"
#include "Grid_RealVect.h"

namespace orchestration {
IntVect::operator RealVect() const {
    return RealVect(Real(vect[0]),Real(vect[1]),Real(vect[2]));
}

// Scalar multiply a vector (c * V)
IntVect operator* (const int c, const IntVect& a) {
    return IntVect(a[0]*c, a[1]*c, a[2]*c);
}

}
