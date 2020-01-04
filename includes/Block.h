#ifndef BLOCK_H__
#define BLOCK_H__

#include <array>
#include <vector>

#include "constants.h"
#include "Grid.h"

class Block {
public:
    Block(void);
    Block(const unsigned int index, Grid* myGrid);
    ~Block(void);

    unsigned int                  index(void) const;
    unsigned int                  nGuardcells(void) const;
    std::array<unsigned int,NDIM> lo(void) const;
    std::array<unsigned int,NDIM> hi(void) const;
    std::array<int,NDIM>          loGC(void) const;
    std::array<int,NDIM>          hiGC(void) const;
    std::array<unsigned int,NDIM> size(void) const;
    std::array<double,NDIM>       deltas(void) const;
    std::vector<double>           coordinates(const int axis) const;
    double***                     dataPtr(void);

    bool isValid(void) const;

private:
    // Element-wise copies fine since the pointer is to a fixed
    // memory allocation managed externally to Block objects.
//    Block& operator=(const Block& rhs);
//    Block(const Block& other);

    unsigned int idx_;
    unsigned int nxb_;
    unsigned int nyb_;
    unsigned int x_idx_;
    unsigned int y_idx_;
    unsigned int nGuard_;
    Grid*        grid_;
};

#endif

