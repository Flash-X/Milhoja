/**
 * \file    TileIterBaseAmrex.h
 *
 * \brief
 *
 */

#ifndef TILEITERBASEAMREX_H__
#define TILEITERBASEAMREX_H__

#include "TileIterBase.h"
#include "TileAmrex.h"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

namespace orchestration {

/**
  * Use in for-loops: 
  *   `for (TileIter ti = grid.buildTileIter(0); ti.isValid(); ++ti)`
  *
  * TODO: implement syntax:
  *   `for (std::shared_ptr<DataItem> tileDesc : grid.buildTileIter(0) )`
  */
class TileIterBaseAmrex : public TileIterBase {
public:
    TileIterBaseAmrex(amrex::MultiFab* mf_in, const unsigned int lev)
        : lev_{lev},
          mfi_{*mf_in} {}

    ~TileIterBaseAmrex() {}

    bool isValid() const override { return mfi_.isValid(); }
    void operator++() override { ++mfi_; }
    std::unique_ptr<Tile> buildCurrentTile() override {
        return std::unique_ptr<Tile>{ new TileAmrex(mfi_,lev_) };
    }

private:
    unsigned int lev_;
    amrex::MFIter mfi_;
};

}

#endif

