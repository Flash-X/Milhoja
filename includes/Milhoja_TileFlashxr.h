#ifndef MILHOJA_TILE_FLASHXR_H__
#define MILHOJA_TILE_FLASHXR_H__

#include <vector>

#include <AMReX.H>
#include <AMReX_Box.H>

#include "Milhoja.h"
#include "Milhoja_RealVect.h"
#include "Milhoja_IntVect.h"
#include "Milhoja_Tile.h"
#include "Milhoja_FlashxrTileRaw.h"

#ifdef RUNTIME_USES_TILEITER
#error "This file should only be compiled if the Runtime class does not invoke a tile iterator"
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
class TileFlashxr
    : public Tile
{
public:
    TileFlashxr(const unsigned int level,
              const int gridIdxOrBlkId,
              const int tileIdx,
              const amrex::Box&& interior,
              const amrex::Box&& dataArray,
		Real*  unkBlkPtr,
              std::vector<Real*>&& fluxBlkPtrs);
    TileFlashxr(const FlashxrTileRawPtrs tP,
			 const FlashxTileRawInts tI,
			 const FlashxTileRawReals tR);
    ~TileFlashxr(void);

    TileFlashxr(TileFlashxr&)                  = delete;
    TileFlashxr(const TileFlashxr&)            = delete;
    TileFlashxr(TileFlashxr&&)                 = delete;
    TileFlashxr& operator=(TileFlashxr&)       = delete;
    TileFlashxr& operator=(const TileFlashxr&) = delete;
    TileFlashxr& operator=(TileFlashxr&&)      = delete;

    // For AMReX, each tile is indexed by (level, gridIndex, tileIndex).
    unsigned int   level(void) const override     { return level_; }
    int            gridIndex(void) const override { return gridIdxOrBlkId_; }
    int            tileIndex(void) const override { return tileIdx_; }

    // Tile metadata
    unsigned int        nCcVariables(void) const override;
    unsigned int        nFluxVariables(void) const override;
    RealVect            deltas(void) const override;
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
    const int                              gridIdxOrBlkId_;  /**< grid index or block ID - for debugging */
    const int                              tileIdx_;  /**< tile index - unused, for debugging */
    const IntVect                          lo_, hi_;
    const IntVect                          loGC_, hiGC_;
    const int                              nCcComp_;
    const int                              nFluxComp_;
    const RealVect                         deltas_;
    Real*                                  unkBlkPtr_;
    std::vector<Real*>               fluxBlkPtrs_;
};

}
#endif

