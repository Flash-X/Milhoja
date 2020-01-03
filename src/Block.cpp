#include <cassert>
#include <stdexcept>

#include "Block.h"

// TODO: The interface of these methods imply that this class can be used with
// different dimensionalities.  However, the implementations are generally fixed
// to two dimensions.

// Construct an intentionally invalid block
Block::Block(void)
    : idx_(0),
      nxb_(0),
      nyb_(0),
      x_idx_(0),
      y_idx_(0),
      nGuard_(0),
      grid_(NULL)
{
    assert(!isValid());
}

// Construct a valid block
Block::Block(const unsigned int index, Grid* myGrid)
    : idx_(index),
      nxb_(0),
      nyb_(0),
      x_idx_(0),
      y_idx_(0),
      nGuard_(0),
      grid_(myGrid)
{
    std::array<unsigned int,NDIM> blockSize = grid_->blockSize();
    nxb_ = blockSize[IAXIS];
    nyb_ = blockSize[JAXIS];

    // TODO: This should come from Grid as well
    std::array<unsigned int,NDIM> shape = grid_->shape();
    x_idx_ = (idx_ % shape[IAXIS]);
    y_idx_ = (idx_ / shape[IAXIS]);

    nGuard_ = grid_->nGuardcells();

    assert(isValid());
}

Block::~Block(void) { }

bool Block::isValid(void) const {
    if (grid_ == NULL) {
        return false;
    }

    std::array<unsigned int,NDIM> shape = grid_->shape();
    unsigned int nBlocks = shape[IAXIS] * shape[JAXIS];

    return (   (idx_ < nBlocks)
            && (nxb_ > 0)
            && (nyb_ > 0));
}

unsigned int Block::index(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::index] Block is invalid");
    }

    return idx_;
}

unsigned int Block::nGuardcells(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::nGuardcells] Block is invalid");
    }

    return nGuard_;
}

std::array<unsigned int,NDIM> Block::lo(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::lo] Block is invalid");
    }

    std::array<unsigned int,NDIM> lo;

    lo[IAXIS] = nxb_ * x_idx_;
    lo[JAXIS] = nyb_ * y_idx_;

    return lo;
}

std::array<unsigned int,NDIM> Block::hi(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::hi] Block is invalid");
    }

    std::array<unsigned int,NDIM> hi;

    hi[IAXIS] = nxb_ * (x_idx_ + 1) - 1;
    hi[JAXIS] = nyb_ * (y_idx_ + 1) - 1;

    return hi;
}

std::array<int,NDIM> Block::loGC(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::loGC] Block is invalid");
    }

    std::array<int,NDIM> lo;

    lo[IAXIS] = nxb_ * x_idx_ - nGuard_;
    lo[JAXIS] = nyb_ * y_idx_ - nGuard_;

    return lo;
}

std::array<int,NDIM> Block::hiGC(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::hiGC] Block is invalid");
    }

    std::array<int,NDIM> hi;

    hi[IAXIS] = nxb_ * (x_idx_ + 1) - 1 + nGuard_;
    hi[JAXIS] = nyb_ * (y_idx_ + 1) - 1 + nGuard_;

    return hi;
}

std::array<unsigned int,NDIM> Block::size(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::size] Block is invalid");
    }

    std::array<unsigned int,NDIM> size;

    size[IAXIS] = nxb_;
    size[JAXIS] = nyb_;

    return size;
}

std::array<double,NDIM> Block::deltas(void) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::deltas] Block is invalid");
    }

    return grid_->deltas(); 
}

std::vector<double> Block::coordinates(const int axis) const {
    if (!isValid()) {
        throw std::runtime_error("[Block::coordinates] Block is invalid");
    }

    if ((axis < 0) || (axis >= NDIM)) {
        throw std::invalid_argument("[Block::coordinates] Invalid axis");
    }

    std::array<double,NDIM>       x0 = grid_->domain(LOW); 
    std::array<int,NDIM>          bLoGC = loGC(); 
    std::array<int,NDIM>          bHiGC = hiGC(); 
    std::array<double,NDIM>       dx = grid_->deltas();
    std::array<unsigned int,NDIM> blockSize = grid_->blockSize();
    std::vector<double>           coords(blockSize[axis] + 2*nGuard_);

    int i0 = bLoGC[axis];
    for (int i=bLoGC[axis]; i<=bHiGC[axis]; ++i) {
        coords[i-i0] = x0[axis] + (i + 0.5)*dx[axis];
    }

    return coords;
}

double** Block::dataPtr(void) {
    if (!isValid()) {
        throw std::runtime_error("[Block::dataPtr] Block is invalid");
    }

    return grid_->dataPtr(idx_);
}

