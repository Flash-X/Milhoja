/**
 * \file    TileIterAmrex.h
 *
 * \brief
 *
 */

#ifndef TILEITERAMREX_H__
#define TILEITERAMREX_H__

#include "TileIter.h"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

namespace orchestration {

class TileIterAmrex : public TileIter {
public:
    TileIterAmrex(amrex::MultiFab* mf_in, const unsigned int lev, const bool use_tiling=false) 
        : TileIter{lev, use_tiling},
          mfi_{*mf_in}
    {
        currentIdx_ = mfi_.tileIndex();
        endIdx_ = currentIdx_ + mfi_.length();
    }

    ~TileIterAmrex() {}

    bool isValid() const override { return mfi_.isValid(); }
    void operator++() override { ++mfi_; currentIdx_++; }
    std::unique_ptr<Tile> buildCurrentTile() override {
        return std::unique_ptr<Tile>{ new Tile(mfi_,lev_) };
    }

private:
    amrex::MFIter mfi_;
};

}

#endif

