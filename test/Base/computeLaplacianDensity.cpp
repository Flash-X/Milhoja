#include "computeLaplacianDensity.h"

#include "Base.h"

void StaticPhysicsRoutines::computeLaplacianDensity(const milhoja::IntVect& lo,
                                                    const milhoja::IntVect& hi,
                                                    milhoja::FArray4D& U,
                                                    milhoja::FArray3D& scratch,
                                                    const milhoja::RealVect& deltas) {
    //$milhoja  "U": {
    //$milhoja&     "RW": [DENS_VAR]
    //$milhoja& }
    
    milhoja::Real   dx_sqr_inv = 1.0 / (deltas.I() * deltas.I());
    milhoja::Real   dy_sqr_inv = 1.0 / (deltas.J() * deltas.J());

    // Compute Laplacian in scratch
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                  scratch(i, j, k) = 
                           (     (  U(i-1, j,   k, DENS_VAR)
                                  + U(i+1, j,   k, DENS_VAR))
                            - 2.0 * U(i,   j,   k, DENS_VAR) ) * dx_sqr_inv
                         + (     (  U(i  , j-1, k, DENS_VAR)
                                  + U(i  , j+1, k, DENS_VAR))
                            - 2.0 * U(i,   j,   k, DENS_VAR) ) * dy_sqr_inv;
             }
        }
    }

    // Overwrite interior of given block with Laplacian result
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, DENS_VAR) = scratch(i, j, k);
             }
        } 
    }
}

