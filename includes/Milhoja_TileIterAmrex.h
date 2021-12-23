/**
 * \file    Milhoja_TileIterAmrex.h
 *
 * \brief
 *
 */

#ifndef MILHOJA_TILE_ITER_AMREX_H__
#define MILHOJA_TILE_ITER_AMREX_H__

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

#include "Milhoja_TileIter.h"
#include "Milhoja_TileAmrex.h"

#ifndef MILHOJA_GRID_AMREX
#error "This file need not be compiled if the AMReX backend isn't used"
#endif

namespace milhoja {

/**
  * \brief Use for iterating over a level with AMReX.
  *
  * TileIterAmrex is a derived class from TileIter with member
  * functions implemented for AMReX. Essentially a wrapper for
  * amrex::MFIter.
  *
  * TODO implement tiling
  */
class TileIterAmrex : public TileIter {
public:
    //! Constructor from a multifab. Creates the MFIter at initialization.
    TileIterAmrex(amrex::MultiFab& mf_in, const unsigned int lev)
        : lev_{lev},
          mfi_{mf_in},
          mfRef_{mf_in} {}

    //! Default destructor.
    ~TileIterAmrex() = default;

    //TODO delete copy constructors

    //! Check if iterator still has tiles left.
    bool isValid() const override { return mfi_.isValid(); }

    //! Advance iterator by one.
    void next() override { ++mfi_; }

    //! Construct Tile for current index.
    std::unique_ptr<Tile> buildCurrentTile() override {
        return std::unique_ptr<Tile>{ new TileAmrex(mfi_,mfRef_,lev_) };
    }

private:
    unsigned int lev_;       //!< Level of iterator.
    amrex::MFIter mfi_;      //!< MFIter that does the iterating.
    amrex::MultiFab& mfRef_; //!< Ref to multifab for appropriate level.
};

}

#endif

