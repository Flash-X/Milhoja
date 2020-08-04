/**
 * \file    TileIterAmrex.h
 *
 * \brief
 *
 */

#ifndef TILEITERAMREX_H__
#define TILEITERAMREX_H__

#include "TileIter.h"
#include "TileAmrex.h"

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

namespace orchestration {

/**
  * Use in for-loops: 
  *   `for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next())`
  *
  */
class TileIterAmrex : public TileIter {
public:
    TileIterAmrex(amrex::MultiFab& mf_in, const unsigned int lev)
        : lev_{lev},
          mfi_{mf_in},
          mfRef_{mf_in} {}

    ~TileIterAmrex() {}

    //TODO delete copy constructors

    bool isValid() const override { return mfi_.isValid(); }
    void next() override { ++mfi_; }
    std::unique_ptr<Tile> buildCurrentTile() override {
        return std::unique_ptr<Tile>{ new TileAmrex(mfi_,mfRef_,lev_) };
    }

private:
    unsigned int lev_;
    amrex::MFIter mfi_;
    amrex::MultiFab& mfRef_;
};

}

#endif

