#include "computeLaplacianEnergy.h"

#include "Base.h"

void StaticPhysicsRoutines::computeLaplacianEnergy(const milhoja::IntVect& lo,
                                                   const milhoja::IntVect& hi,
                                                   milhoja::FArray4D& U,
                                                   milhoja::FArray4D& scratch,
                                                   const milhoja::RealVect& deltas) {
    milhoja::Real   dx_sqr_inv = 1.0 / (deltas.I() * deltas.I());
    milhoja::Real   dy_sqr_inv = 1.0 / (deltas.J() * deltas.J());

    // OFFLINE TOOLCHAIN - Place parallelization directive/hints here
    // Compute Laplacian in scratch
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                  scratch(i, j, k, 0) = 
                           (     (  U(i-1, j,   k, ENER_VAR)
                                  + U(i+1, j,   k, ENER_VAR))
                            - 2.0 * U(i,   j,   k, ENER_VAR) ) * dx_sqr_inv
                         + (     (  U(i  , j-1, k, ENER_VAR)
                                  + U(i  , j+1, k, ENER_VAR))
                            - 2.0 * U(i,   j,   k, ENER_VAR) ) * dy_sqr_inv;
             }
        }
    }

    // OFFLINE TOOLCHAIN - Place parallelization directive/hints here
    // Overwrite interior of given block with Laplacian result
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, ENER_VAR) = scratch(i, j, k, 0);
             }
        } 
    }
}

