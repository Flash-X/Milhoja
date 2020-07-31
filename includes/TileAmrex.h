#ifndef TILEAMREX_H__
#define TILEAMREX_H__

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include "Grid_IntVect.h"

#include "Tile.h"

namespace orchestration {

/**
  * Derived class from Tile.
  */
class TileAmrex : public Tile {
public:
    TileAmrex(amrex::MFIter& itor, const unsigned int level);
    ~TileAmrex(void);

    TileAmrex(TileAmrex&&);
    TileAmrex& operator=(TileAmrex&&);

    bool             isNull(void) const override;

    IntVect          lo(void) const override;
    IntVect          hi(void) const override;

    IntVect          loGC(void) const override;
    IntVect          hiGC(void) const override;

    FArray4D         data(void) override;
    Real*            dataPtr(void) override;

private:
    // TODO Remove this once AMReX is extracted from Grid and Tile base classes
    amrex::MultiFab&   unk_;

    // Limit all copies as much as possible
    TileAmrex(const TileAmrex&) = delete;
    TileAmrex(TileAmrex&) = delete;
    TileAmrex& operator=(TileAmrex&) = delete;
    TileAmrex& operator=(const TileAmrex&) = delete;
};

}

#endif

