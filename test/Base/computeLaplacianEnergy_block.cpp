#include "computeLaplacianEnergy_block.h"

#include "Flash.h"
#include "Grid.h"
#include "Tile.h"

void ThreadRoutines::computeLaplacianEnergy_block(const int tId, void* dataItem) {
    Tile* tileDesc = static_cast<Tile*>(dataItem);

    amrex::MultiFab&    unk = Grid::instance().unk();
    amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];
    amrex::Array4<amrex::Real> const&   f = fab.array();

    // TODO: We should have scratch preallocated outside this routine.  It would
    // either be setup by the operation or available through a memory manager
    // and its location communicated as a pointer in a given data packet.
    amrex::FArrayBox                    fabBuffer(tileDesc->interior());
    amrex::Array4<amrex::Real> const&   buffer = fabBuffer.array();

    amrex::XDim3  deltas = tileDesc->deltas();

    amrex::Real   f_i     = 0.0;
    amrex::Real   f_x_im1 = 0.0;
    amrex::Real   f_x_ip1 = 0.0;
    amrex::Real   f_y_im1 = 0.0;
    amrex::Real   f_y_ip1 = 0.0;

    amrex::Real   dx_sqr_inv = 1.0 / (deltas.x * deltas.x);
    amrex::Real   dy_sqr_inv = 1.0 / (deltas.y * deltas.y);

    // Compute Laplacian in buffer
    const amrex::Dim3 lo = tileDesc->lo();
    const amrex::Dim3 hi = tileDesc->hi();
    for     (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
              f_i     = f(i,   j,   lo.z, ENER_VAR_C);
              f_x_im1 = f(i-1, j,   lo.z, ENER_VAR_C);
              f_x_ip1 = f(i+1, j,   lo.z, ENER_VAR_C);
              f_y_im1 = f(i,   j-1, lo.z, ENER_VAR_C);
              f_y_ip1 = f(i,   j+1, lo.z, ENER_VAR_C);
              buffer(i, j, lo.z, 0) = 
                    ((f_x_im1 + f_x_ip1) - 2.0*f_i) * dx_sqr_inv
                  + ((f_y_im1 + f_y_ip1) - 2.0*f_i) * dy_sqr_inv;
         }
    }

    // Overwrite interior of given block with Laplacian result
    // TODO: In the case of a data packet, we could have the input data given as
    // a pointer to CC1 and directly write the result to CC2.  When copying the
    // data back to UNK, we copy from CC2 and ignore CC1.  Therefore, this copy
    // would be unnecessary.
    for     (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
            f(i, j, lo.z, ENER_VAR_C) = buffer(i, j, lo.z, 0);
         }
    } 
}

