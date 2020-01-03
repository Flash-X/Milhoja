#include <stdexcept>

#include "Grid.h"

// TODO: The interface of these methods imply that this class can be used with
// different dimensionalities.  However, the implementations are generally fixed
// to two dimensions.

Grid::Grid(const double xMin, const double xMax,
           const double yMin, const double yMax,
           const unsigned int nCellsPerBlockX, 
           const unsigned int nCellsPerBlockY, 
           const unsigned int nBlocksX,
           const unsigned int nBlocksY, 
           const unsigned int nGuard)
    : xMin_(xMin),
      xMax_(xMax),
      yMin_(yMin),
      yMax_(yMax),
      nxb_(nCellsPerBlockX),
      nyb_(nCellsPerBlockY),
      nBlocksX_(nBlocksX),
      nBlocksY_(nBlocksY),
      nGuard_(nGuard),
      blocks_(NULL)
{
    // TODO: Error check input parameters

    unsigned int nBlocks = nBlocksX_ * nBlocksY_;
    unsigned int nCellsX = nxb_ + 2*nGuard_;
    unsigned int nCellsY = nyb_ + 2*nGuard_;

    // Allocate each block as a contiguous region of memory, but
    // allow for blocks residing in different parts of memory.
    blocks_ = new double**[nBlocks];
    for     (unsigned int n=0; n<nBlocks; ++n) {
        blocks_[n]    = new double*[nCellsX];
        blocks_[n][0] = new double[nCellsX * nCellsY];
        for (unsigned int i=1; i<nCellsX; ++i) {
            blocks_[n][i] = blocks_[n][i-1] + nCellsY;
        }
    }
}

Grid::~Grid(void) {
    unsigned int nBlocks = nBlocksX_ * nBlocksY_;

    for (unsigned int n=0; n<nBlocks; ++n) {
        delete [] blocks_[n][0];
        delete [] blocks_[n];
    }
    delete [] blocks_;
}

unsigned int Grid::nGuardcells(void) const { 
    return nGuard_;
}

std::array<unsigned int,NDIM> Grid::shape(void) const {
    std::array<unsigned int,NDIM> out;

    out[IAXIS] = nBlocksX_;
    out[JAXIS] = nBlocksY_;

    return out;
}

std::array<unsigned int,NDIM> Grid::blockSize(void) const {
    std::array<unsigned int,NDIM> out;

    out[IAXIS] = nxb_;
    out[JAXIS] = nyb_;

    return out;
}

std::array<double,NDIM> Grid::domain(const int point) const {
    std::array<double,NDIM>  pt;

    if        (point == LOW) {
        pt[IAXIS] = xMin_;
        pt[JAXIS] = yMin_;
    } else if (point == HIGH) {
        pt[IAXIS] = xMax_;
        pt[JAXIS] = yMax_;
    } else {
        throw std::invalid_argument("[Grid::domain] Invalid point");
    }

    return pt;
}

std::array<double,NDIM> Grid::deltas(void) const {
    std::array<double,NDIM> dx;

    unsigned int nCellsX = nxb_ * nBlocksX_;
    unsigned int nCellsY = nyb_ * nBlocksY_;

    dx[IAXIS] = (xMax_ - xMin_) / ((double)nCellsX);
    dx[JAXIS] = (yMax_ - yMin_) / ((double)nCellsY);

    return dx;
}

double** Grid::dataPtr(const unsigned int idx) {
    unsigned int nBlocks = nBlocksX_ * nBlocksY_;

    if (idx >= nBlocks) {
        throw std::invalid_argument("[Grid::dataPtr] Invalid index");
    }

    return blocks_[idx];
}

