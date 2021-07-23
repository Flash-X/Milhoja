#include "DataShape.h"
#include <cstring>
#include <cassert>

namespace orchestration {

DataShape::DataShape(unsigned int n, unsigned int d)
      : n_{n},
        d_{d},
        ls_{nullptr},
        pad_{nullptr},
        part_{nullptr}
{
    // allocate and setup list of shapes (initialize to zero with `()`)
    ls_    = new std::size_t*[n_+1]();
    ls_[0] = new std::size_t[n_*d_]();
    for (unsigned int i=1; i<n_; ++i) {
        ls_[i] = ls_[i-1] + d_;
    }
    ls_[n_] = nullptr;
    // allocate padding
    pad_ = new std::size_t[n_+1]();
    pad_[n_] = 0;
    // allocate partitions
    part_ = new unsigned int[n_+1]();
    part_[n_] = 0;
}

DataShape::~DataShape(void) {
    delete[] ls_;
    delete[] pad_;
    delete[] part_;
}

void  DataShape::set(const std::size_t* entries) {
    std::memcpy(ls_[0], entries, nEntries());
}

void  DataShape::setPadding(const std::size_t* pad) {
    std::memcpy(pad_, pad, n_);
}

void  DataShape::setPartition(const unsigned int* part) {
    std::memcpy(part_, part, n_);
}

std::size_t DataShape::size(void) const {
    std::size_t s, size=0;

    // check self
    assert(ls_);
    assert(pad_);

    // sum over variables
    for (unsigned int i=0; i<n_; i++) {
        // multiply over shape dimensions
        s = ls_[i][0];
        for (unsigned int j=1; j<d_; j++) {
            if (0 < ls_[i][j]) {
                s *= ls_[i][j];
            }
        }
        size += s + pad_[i];
    }

    return size;
}

std::size_t DataShape::sizePartition(unsigned int partIdx) const {
    std::size_t s, size=0;

    // check self
    assert(ls_);
    assert(pad_);
    assert(part_);

    // sum over variables
    for (unsigned int i=0; i<n_; i++) {
        // skip if partition does not match
        if (part_[i] != partIdx) {
            continue;
        }
        // multiply over shape dimensions
        s = ls_[i][0];
        for (unsigned int j=1; j<d_; j++) {
            if (0 < ls_[i][j]) {
                s *= ls_[i][j];
            }
        }
        size += s + pad_[i];
    }

    return size;
}

std::size_t DataShape::sizeVariable(unsigned int varIdx) const {
    std::size_t s, size=0;

    // check self
    assert(ls_);
    assert(pad_);
    // check input
    if (n_ <= varIdx) throw std::out_of_range("[DataShape::atPart] Variable index out of range");

    // multiply over shape dimensions
    s = ls_[varIdx][0];
    for (unsigned int j=1; j<d_; j++) {
        if (0 < ls_[varIdx][j]) {
            s *= ls_[varIdx][j];
        }
    }
    size += s + pad_[varIdx];

    return size;
}

}

