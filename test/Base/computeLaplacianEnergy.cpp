#include "computeLaplacianEnergy.h"

#include "Flash.h"

void StaticPhysicsRoutines::computeLaplacianEnergy(const orchestration::IntVect& lo,
                                                   const orchestration::IntVect& hi,
                                                   orchestration::FArray4D& f,
                                                   orchestration::FArray4D& scratch,
                                                   const orchestration::RealVect& deltas) {
    using namespace orchestration;

    Real   dx_sqr_inv = 1.0 / (deltas.I() * deltas.I());
    Real   dy_sqr_inv = 1.0 / (deltas.J() * deltas.J());

    // OFFLINE TOOLCHAIN - Place parallelization directive/hints here
    // Compute Laplacian in scratch
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                  scratch(i, j, k, 0) = 
                           (     (  f(i-1, j,   k, ENER_VAR_C)
                                  + f(i+1, j,   k, ENER_VAR_C))
                            - 2.0 * f(i,   j,   k, ENER_VAR_C) ) * dx_sqr_inv
                         + (     (  f(i  , j-1, k, ENER_VAR_C)
                                  + f(i  , j+1, k, ENER_VAR_C))
                            - 2.0 * f(i,   j,   k, ENER_VAR_C) ) * dy_sqr_inv;
             }
        }
    }

    // OFFLINE TOOLCHAIN - Place parallelization directive/hints here
    // Overwrite interior of given block with Laplacian result
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                f(i, j, k, ENER_VAR_C) = scratch(i, j, k, 0);
             }
        } 
    }
}

