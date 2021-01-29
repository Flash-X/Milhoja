/**
 * \file    TileIterAmrex.h
 *
 * \brief
 *
 */

#ifndef TILEITERAMREX_H__
#define TILEITERAMREX_H__

#ifdef USE_THREADED_DISTRIBUTOR
#include <omp.h>
#include <cstdio>
#endif

#include "TileIter.h"
#include "TileAmrex.h"

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

namespace orchestration {

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
          mfRef_{mf_in} {
#ifdef USE_THREADED_DISTRIBUTOR
    int tId = omp_get_thread_num();
    printf("[Thread %d] TileIterAmrex created its own MFIter\n", tId);
#endif
}

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

