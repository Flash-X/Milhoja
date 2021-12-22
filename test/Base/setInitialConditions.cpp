#include "setInitialConditions.h"

#include "Base.h"

void StaticPhysicsRoutines::setInitialConditions(const milhoja::IntVect& loGC,
                                                 const milhoja::IntVect& hiGC,
                                                 const milhoja::FArray1D& xCoords,
                                                 const milhoja::FArray1D& yCoords,
                                                 milhoja::FArray4D& U) {
    milhoja::Real    x = 0.0;
    milhoja::Real    y = 0.0;
    for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
        for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
            y = yCoords(j);
            for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                x = xCoords(i); 

                // PROBLEM ONE
                //  Approximated exactly by second-order discretized Laplacian
                U(i, j, k, DENS_VAR) =   3.0*x*x*x +     x*x + x 
                                       - 2.0*y*y*y - 1.5*y*y + y
                                       + 5.0;
                // PROBLEM TWO
                //  Approximation is not exact and we know the error term exactly
                U(i, j, k, ENER_VAR) =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                                       -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                                       + 1.0;
            }
        }
    }
}

