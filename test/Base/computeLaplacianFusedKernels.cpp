#include "computeLaplacianFused.h"

#include "Tile.h"

#include "Flash.h"

void StaticPhysicsRoutines::computeLaplacianFusedKernels(const orchestration::IntVect& lo,
                                                         const orchestration::IntVect& hi,
                                                         orchestration::FArray4D& U,
                                                         orchestration::FArray4D& scratch,
                                                         const orchestration::RealVect& deltas) {
    using namespace orchestration;

    Real   dx_sqr_inv = 1.0 / (deltas.I() * deltas.I());
    Real   dy_sqr_inv = 1.0 / (deltas.J() * deltas.J());

    // Assume OFFLINE TOOLCHAIN determined that the K loop was not necessary
    // and that it determined to fuse loop nests here and below
    for     (int j=lo.J(); j<=hi.J(); ++j) {
        for (int i=lo.I(); i<=hi.I(); ++i) {
              scratch(i, j, 0, DENS_VAR) = 
                       (     (  U(i-1, j,   0, DENS_VAR)
                              + U(i+1, j,   0, DENS_VAR))
                        - 2.0 * U(i,   j,   0, DENS_VAR) ) * dx_sqr_inv
                     + (     (  U(i  , j-1, 0, DENS_VAR)
                              + U(i  , j+1, 0, DENS_VAR))
                        - 2.0 * U(i,   j,   0, DENS_VAR) ) * dy_sqr_inv;

              scratch(i, j, 0, ENER_VAR) = 
                       (     (  U(i-1, j,   0, ENER_VAR)
                              + U(i+1, j,   0, ENER_VAR))
                        - 2.0 * U(i,   j,   0, ENER_VAR) ) * dx_sqr_inv
                     + (     (  U(i  , j-1, 0, ENER_VAR)
                              + U(i  , j+1, 0, ENER_VAR))
                        - 2.0 * U(i,   j,   0, ENER_VAR) ) * dy_sqr_inv;
         }
    }

    for     (int j=lo.J(); j<=hi.J(); ++j) {
        for (int i=lo.I(); i<=hi.I(); ++i) {
            U(i, j, 0, DENS_VAR) = scratch(i, j, 0, DENS_VAR);
            U(i, j, 0, ENER_VAR) = scratch(i, j, 0, ENER_VAR);
         }
    } 
}

