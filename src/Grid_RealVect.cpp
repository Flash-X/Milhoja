#include "Grid_RealVect.h"
#include "Grid_IntVect.h"

namespace orchestration {

RealVect::operator IntVect() const {
    return IntVect(int(vect[0]),int(vect[1]),int(vect[2]));
}

// Scalar multiply a vector (c * V)
RealVect operator* (const Real c, const RealVect& a) {
   return RealVect(a[0]*c, a[1]*c, a[2]*c);
}

}
