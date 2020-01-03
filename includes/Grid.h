#ifndef GRID_H__
#define GRID_H__

#include <array>

#include "constants.h"

class Grid {
public:
    Grid(const double xMin, const double xMax,
         const double yMin, const double yMax,
         const unsigned int nCellsPerBlockX, 
         const unsigned int nCellsPerBlockY, 
         const unsigned int nBlocksX,
         const unsigned int nBlocksY,
         const unsigned int nGuard);
    ~Grid(void);

    unsigned int                  nGuardcells(void) const;
    std::array<unsigned int,NDIM> shape(void) const;
    std::array<unsigned int,NDIM> blockSize(void) const;
    std::array<double,NDIM>       domain(const int point) const;
    std::array<double,NDIM>       deltas(void) const;

    // TODO: This could be a protected method.  If we make the Blocks
    //       a friend of this class, then they and they alone can get
    //       their pointer to a FAB through this means.
    // The pointer is to the FAB interior + GCs.  The zero-based array 
    // indices are to the first GC on the low faces of the blocks.
    double** dataPtr(const unsigned int idx);

private:
    // Disallow copying due to dynamic memory allocation of multifab
    Grid& operator=(const Grid& rhs);
    Grid(const Grid& other);

    double    xMin_;
    double    xMax_;
    double    yMin_;
    double    yMax_;

    unsigned int nxb_;
    unsigned int nyb_;
    unsigned int nBlocksX_;
    unsigned int nBlocksY_;
    unsigned int nGuard_;

    // Our multifab is an array of 2D block data arrays (the FABs)
    // The zero-based block index is the index into this array.
    double*** blocks_;
};

#endif

