#ifndef MILHOJA_TILE_AMREX_H__
#define MILHOJA_TILE_AMREX_H__

#include <vector>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

#include "Milhoja.h"
#include "Milhoja_IntVect.h"
#include "Milhoja_Tile.h"

#ifndef MILHOJA_GRID_AMREX
#error "This file need not be compiled if the AMReX backend isn't used"
#endif

namespace milhoja {

/**
 * \brief Fully-implemented Tile class for use with AMReX.
 *
 * \todo lo/hi     -> interiorBox(lo, hi)
 * \todo loGC/hiGC -> dataArrayBox(lo, hi)
 * \todo need grownTileBox as well.
 *
 * Contains a reference to the multifab for the appropriate level
 * so pointers into it can be returned.
 */
class TileAmrex
    : public Tile
{
public:
    TileAmrex(const unsigned int level,
              const int gridIdx,
              const int tileIdx,
              const amrex::Box&& interior,
              const amrex::Box&& dataArray,
              amrex::FArrayBox& unkFab,
              std::vector<amrex::FArrayBox*>&& fluxFabs);
    ~TileAmrex(void);

    TileAmrex(TileAmrex&)                  = delete;
    TileAmrex(const TileAmrex&)            = delete;
    TileAmrex(TileAmrex&&)                 = delete;
    TileAmrex& operator=(TileAmrex&)       = delete;
    TileAmrex& operator=(const TileAmrex&) = delete;
    TileAmrex& operator=(TileAmrex&&)      = delete;

    // For AMReX, each tile is indexed by (level, gridIndex, tileIndex).
    unsigned int   level(void) const override     { return level_; }
    int            gridIndex(void) const override { return gridIdx_; }
    int            tileIndex(void) const override { return tileIdx_; }

    // Tile metadata
    unsigned int        nCcVariables(void) const override;
    unsigned int        nFluxVariables(void) const override;
    IntVect             lo(void) const override;
    IntVect             hi(void) const override;
    IntVect             loGC(void) const override;
    IntVect             hiGC(void) const override;
    FArray4D            data(void) override;
    Real*               dataPtr(void) override;
    FArray4D            fluxData(const unsigned int dir) override;
    std::vector<Real*>  fluxDataPtrs(void) override;

private:
    const unsigned int                     level_;    /**< 0-based level of Tile */
    const int                              gridIdx_;  /**< Index into multifab */
    const int                              tileIdx_;  /**< Index into multifab */
    const amrex::Box                       interiorBox_;
    const amrex::Box                       dataArrayBox_;
    amrex::FArrayBox&                      unkFab_;
    const std::vector<amrex::FArrayBox*>   fluxFabs_;
};

}
#endif

