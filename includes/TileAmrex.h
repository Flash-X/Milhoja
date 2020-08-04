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
    TileAmrex(amrex::MFIter& itor, amrex::MultiFab& unkRef, const unsigned int level);
    ~TileAmrex(void);

    TileAmrex(TileAmrex&&) = delete;
    TileAmrex& operator=(TileAmrex&&) = delete;
    TileAmrex(const TileAmrex&) = delete;
    TileAmrex(TileAmrex&) = delete;
    TileAmrex& operator=(TileAmrex&) = delete;
    TileAmrex& operator=(const TileAmrex&) = delete;

    // Overrides to pure virtual functions
    bool         isNull(void) const override;
    int          gridIndex(void) const override { return gridIdx_; }
    unsigned int level(void) const override { return level_; }

    IntVect          lo(void) const override;
    IntVect          hi(void) const override;
    IntVect          loGC(void) const override;
    IntVect          hiGC(void) const override;

    FArray4D         data(void) override;
    Real*            dataPtr(void) override;

protected:
    unsigned int  level_;
    int           gridIdx_;
    amrex::Box*   interior_;
    amrex::Box*   GC_;

    // TODO Remove this once AMReX is extracted from Grid and Tile base classes?
    amrex::MultiFab&   unkRef_;
};

}

#endif

