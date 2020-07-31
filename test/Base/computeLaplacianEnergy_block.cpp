#include "computeLaplacianEnergy_block.h"

#include "Grid.h"
#include "Tile.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"

#include "Flash.h"

void ThreadRoutines::computeLaplacianEnergy_block(const int tId, void* dataItem) {
    using namespace orchestration;

    Tile* tileDesc = static_cast<Tile*>(dataItem);

    Grid&  grid = Grid::instance();

    const IntVect   lo     = tileDesc->lo();
    const IntVect   hi     = tileDesc->hi();
    const RealVect  deltas = tileDesc->deltas();
    FArray4D        f      = tileDesc->data();

    // TODO: This exposes low-level data storage details to the PUD.  Hide!
    // TODO: This would need to be done in an NDIM-specific way
    // TODO: We should have scratch preallocated outside this routine.  It would
    // either be setup by the operation or available through a memory manager
    // and its location communicated as a pointer in a given data packet.
    Real  buffer1D[   (hi[Axis::I] - lo[Axis::I] + 1)
                    * (hi[Axis::J] - lo[Axis::J] + 1)
                    *  NUNKVAR];
    FArray4D  buffer{buffer1D, lo, hi, NUNKVAR};

    Real   f_i     = 0.0;
    Real   f_x_im1 = 0.0;
    Real   f_x_ip1 = 0.0;
    Real   f_y_im1 = 0.0;
    Real   f_y_ip1 = 0.0;

    Real   dx_sqr_inv = 1.0 / (deltas[Axis::I] * deltas[Axis::I]);
    Real   dy_sqr_inv = 1.0 / (deltas[Axis::J] * deltas[Axis::J]);

    // Compute Laplacian in buffer
 //   for         (int k = lo[Axis::K]; k <= hi[Axis::K]; ++k) {
        for     (int j = lo[Axis::J]; j <= hi[Axis::J]; ++j) {
            for (int i = lo[Axis::I]; i <= hi[Axis::I]; ++i) {
                f_i     = f(i,   j,   ENER_VAR_C);
                f_x_im1 = f(i-1, j,   ENER_VAR_C);
                f_x_ip1 = f(i+1, j,   ENER_VAR_C);
                f_y_im1 = f(i,   j-1, ENER_VAR_C);
                f_y_ip1 = f(i,   j+1, ENER_VAR_C);
                buffer(i, j, 0) = 
                      ((f_x_im1 + f_x_ip1) - 2.0*f_i) * dx_sqr_inv
                    + ((f_y_im1 + f_y_ip1) - 2.0*f_i) * dy_sqr_inv;
            }
        }
//    }

    // Overwrite interior of given block with Laplacian result
    // TODO: In the case of a data packet, we could have the input data given as
    // a pointer to CC1 and directly write the result to CC2.  When copying the
    // data back to UNK, we copy from CC2 and ignore CC1.  Therefore, this copy
    // would be unnecessary.
//    for         (int k = lo[Axis::K]; k <= hi[Axis::K]; ++k) {
        for     (int j = lo[Axis::J]; j <= hi[Axis::J]; ++j) {
            for (int i = lo[Axis::I]; i <= hi[Axis::I]; ++i) {
                f(i, j, ENER_VAR_C) = buffer(i, j, 0);
            }
        }
//    } 
}

