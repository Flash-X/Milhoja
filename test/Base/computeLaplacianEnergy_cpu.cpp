#include "computeLaplacianEnergy_cpu.h"

#include "Flash.h"
#include "Grid.h"

void ThreadRoutines::computeLaplacianEnergy_cpu(const int tId, Tile* tileDesc) {
    amrex::MultiFab&    unk = Grid::instance()->unk();
    amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];
    amrex::Array4<amrex::Real> const&   f = fab.array();

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
    for     (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
            f(i, j, lo.z, ENER_VAR_C) = buffer(i, j, lo.z, 0);
         }
    } 
}

