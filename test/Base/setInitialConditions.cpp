#include "setInitialConditions.h"

#include "Test.h"

void StaticPhysicsRoutines::setInitialConditions(const orchestration::IntVect& loGC,
                                                 const orchestration::IntVect& hiGC,
                                                 const orchestration::FArray1D& xCoords,
                                                 const orchestration::FArray1D& yCoords,
                                                 orchestration::FArray4D& U) {
    using namespace orchestration;

    Real    x = 0.0;
    Real    y = 0.0;
    for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
        for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
            y = yCoords(j);
            for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                x = xCoords(i); 

                // PROBLEM ONE
                //  Approximated exactly by second-order discretized Laplacian
                U(i, j, k, DENS_VAR_C) =   3.0*x*x*x +     x*x + x 
                                         - 2.0*y*y*y - 1.5*y*y + y
                                         + 5.0;
                // PROBLEM TWO
                //  Approximation is not exact and we know the error term exactly
                U(i, j, k, ENER_VAR_C) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                                         -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                         + 1.0;
            }
        }
    }
}

