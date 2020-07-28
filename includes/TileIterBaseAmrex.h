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

class TileIterBaseAmrex : public TileIterBase {
public:
    TileIterBaseAmrex(amrex::MultiFab* mf_in, const unsigned int lev)
        : TileIterBase{lev},
          mfi_{*mf_in}
    {
        currentIdx_ = mfi_.tileIndex();
        endIdx_ = currentIdx_ + mfi_.length();
    }

    ~TileIterBaseAmrex() {}

    bool isValid() const override { return mfi_.isValid(); }
    void operator++() override { ++mfi_; currentIdx_++; }
    std::unique_ptr<Tile> buildCurrentTile() override {
        return std::unique_ptr<TileAmrex>{ new TileAmrex(mfi_,lev_) };
    }

private:
    amrex::MFIter mfi_;
};

}

#endif

