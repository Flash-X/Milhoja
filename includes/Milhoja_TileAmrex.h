#ifndef MILHOJA_TILE_AMREX_H__
#define MILHOJA_TILE_AMREX_H__

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

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
 * Contains a reference to the multifab for the appropriate level
 * so pointers into it can be returned.
 */
class TileAmrex
    : public Tile
{
public:
    TileAmrex(amrex::MFIter& itor, amrex::MultiFab& unkRef,
              const unsigned int level);
    ~TileAmrex(void);

    TileAmrex(TileAmrex&&) = delete;
    TileAmrex& operator=(TileAmrex&&) = delete;
    TileAmrex(const TileAmrex&) = delete;
    TileAmrex(TileAmrex&) = delete;
    TileAmrex& operator=(TileAmrex&) = delete;
    TileAmrex& operator=(const TileAmrex&) = delete;

    // Overrides to pure virtual functions
    bool           isNull(void) const override;
    int            gridIndex(void) const override { return gridIdx_; }
    unsigned int   level(void) const override { return level_; }

    IntVect        lo(void) const override;
    IntVect        hi(void) const override;
    IntVect        loGC(void) const override;
    IntVect        hiGC(void) const override;

    FArray4D       data(void) override;
    Real*          dataPtr(void) override;

private:
    unsigned int  level_;    /**< 0-based level of Tile */
    int           gridIdx_;  /**< Index into multifab */
    amrex::Box*   interior_; /**< Box of tile */
    amrex::Box*   GC_;       /**< Grown box of tile */

    // TODO Remove this once AMReX is extracted from Grid and Tile base classes?
    amrex::MultiFab&   unkRef_; /**< Ref to multifab with physical data */
};


}
#endif

