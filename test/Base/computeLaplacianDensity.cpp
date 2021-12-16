#include "computeLaplacianDensity.h"

#include "Base.h"

void StaticPhysicsRoutines::computeLaplacianDensity(const orchestration::IntVect& lo,
                                                    const orchestration::IntVect& hi,
                                                    orchestration::FArray4D& U,
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
                           (     (  U(i-1, j,   k, DENS_VAR)
                                  + U(i+1, j,   k, DENS_VAR))
                            - 2.0 * U(i,   j,   k, DENS_VAR) ) * dx_sqr_inv
                         + (     (  U(i  , j-1, k, DENS_VAR)
                                  + U(i  , j+1, k, DENS_VAR))
                            - 2.0 * U(i,   j,   k, DENS_VAR) ) * dy_sqr_inv;
             }
        }
    }

    // OFFLINE TOOLCHAIN - Place parallelization directive/hints here
    // Overwrite interior of given block with Laplacian result
    // TODO: In the case of a data packet, we could have the input data given as
    // a pointer to CC1 and directly write the result to CC2.  When copying the
    // data back to UNK, we copy from CC2 and ignore CC1.  Therefore, this copy
    // would be unnecessary.
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                U(i, j, k, DENS_VAR) = scratch(i, j, k, 0);
             }
        } 
    }
}

