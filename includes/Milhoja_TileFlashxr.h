#ifndef MILHOJA_TILE_FLASHXR_H__
#define MILHOJA_TILE_FLASHXR_H__

#include <vector>

#include "Milhoja.h"
#include "Milhoja_RealVect.h"
#include "Milhoja_IntVect.h"
#include "Milhoja_Tile.h"
#include "Milhoja_FlashxrTileRaw.h"

#ifdef RUNTIME_MUST_USE_TILEITER
#error "This file should only be compiled if the Runtime class need not invoke a tile iterator"
#endif

namespace milhoja {

class TileFlashxr
    : public Tile
{
public:
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
    const unsigned int               level_;    /**< 0-based level of Tile */
    const int                        gridIdxOrBlkId_;  /**< grid index or block ID - for debugging */
    const int                        tileIdx_;  /**< tile index - unused, for debugging */
    const IntVect                    lo_, hi_;
    const IntVect                    loGC_, hiGC_;
    const int                        nCcComp_;
    const int                        nFluxComp_;
    const RealVect                   deltas_;
    Real*                            unkBlkPtr_;
    std::vector<Real*>               fluxBlkPtrs_;
};

}
#endif

