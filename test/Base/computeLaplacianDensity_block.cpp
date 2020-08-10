#include "computeLaplacianDensity_block.h"

#include "Grid.h"
#include "Tile.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"

#include "Flash.h"

void ThreadRoutines::computeLaplacianDensity_block(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = static_cast<Tile*>(dataItem);

    Grid&  grid = Grid::instance();

    const IntVect   lo     = tileDesc->lo();
    const IntVect   hi     = tileDesc->hi();
    const RealVect  deltas = tileDesc->deltas();
    FArray4D        f      = tileDesc->data();

    // TODO: We should have scratch preallocated outside this routine.  It would
    // either be setup by the operation or available through a memory manager
    // and its location communicated as a pointer in a given data packet.
    FArray4D  scratch = FArray4D::buildScratchArray4D(lo, hi, 1);

    Real   f_i     = 0.0;
    Real   f_x_im1 = 0.0;
    Real   f_x_ip1 = 0.0;
    Real   f_y_im1 = 0.0;
    Real   f_y_ip1 = 0.0;

    Real   dx_sqr_inv = 1.0 / (deltas[Axis::I] * deltas[Axis::I]);
    Real   dy_sqr_inv = 1.0 / (deltas[Axis::J] * deltas[Axis::J]);

    // Compute Laplacian in scratch
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            for (int i = lo.I(); i <= hi.I(); ++i) {
                  f_i     = f(i,   j,   k, DENS_VAR_C);
                  f_x_im1 = f(i-1, j,   k, DENS_VAR_C);
                  f_x_ip1 = f(i+1, j,   k, DENS_VAR_C);
                  f_y_im1 = f(i,   j-1, k, DENS_VAR_C);
                  f_y_ip1 = f(i,   j+1, k, DENS_VAR_C);
                  scratch(i, j, k, 0) = 
                        ((f_x_im1 + f_x_ip1) - 2.0*f_i) * dx_sqr_inv
                      + ((f_y_im1 + f_y_ip1) - 2.0*f_i) * dy_sqr_inv;
             }
        }
    }

    // Overwrite interior of given block with Laplacian result
    // TODO: In the case of a data packet, we could have the input data given as
    // a pointer to CC1 and directly write the result to CC2.  When copying the
    // data back to UNK, we copy from CC2 and ignore CC1.  Therefore, this copy
    // would be unnecessary.
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            for (int i = lo.I(); i <= hi.I(); ++i) {
                f(i, j, k, DENS_VAR_C) = scratch(i, j, k, 0);
             }
        } 
    }
}

