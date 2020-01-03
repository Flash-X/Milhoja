#include <cassert>
#include <stdexcept>

#include "BlockIterator.h"

BlockIterator::BlockIterator(Grid* myGrid)
      : grid_(myGrid),
        idx_(0)
{
    clear();

    assert(isValid());
}

BlockIterator::~BlockIterator(void) {
}

void BlockIterator::clear(void) {
    idx_ = 0;
}

bool BlockIterator::isValid(void) const {
    if (grid_ == NULL) {
        return false;
    }

    std::array<unsigned int,NDIM> shape = grid_->shape();

    return (idx_ < shape[IAXIS]*shape[JAXIS]);
}

Block BlockIterator::currentBlock(void) {
    if (!isValid()) {
        throw std::runtime_error("BlockIterator is invalid");
    }

    return Block(idx_, grid_);
}

void BlockIterator::next(void) {
    ++idx_;
}

