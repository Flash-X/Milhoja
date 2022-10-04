#ifndef MILHOJA_TILE_ITER_AMREX_H__
#define MILHOJA_TILE_ITER_AMREX_H__

#include <vector>

#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include "Milhoja.h"
#include "Milhoja_Tile.h"
#include "Milhoja_TileIter.h"

#ifndef MILHOJA_AMREX_GRID_BACKEND
#error "This file need not be compiled if the AMReX backend isn't used"
#endif

namespace milhoja {

/**
  * \brief Use for iterating over a level with AMReX.
  *
  * TileIterAmrex is a derived class from TileIter with member functions
  * implemented for AMReX. Essentially a wrapper for amrex::MFIter.
  *
  * \todo Test tiling & permit its use
  */
class TileIterAmrex : public TileIter {
public:
    TileIterAmrex(amrex::MultiFab& unk,
                  std::vector<amrex::MultiFab>& fluxes,
                  const unsigned int level);
    ~TileIterAmrex();

    TileIterAmrex(TileIterAmrex&)                  = delete;
    TileIterAmrex(const TileIterAmrex&)            = delete;
    TileIterAmrex(TileIterAmrex&&)                 = delete;
    TileIterAmrex& operator=(TileIterAmrex&)       = delete;
    TileIterAmrex& operator=(const TileIterAmrex&) = delete;
    TileIterAmrex& operator=(TileIterAmrex&&)      = delete;

    //! Check if iterator still has tiles left.
    bool isValid() const override { return mfi_.isValid(); }

    //! Advance iterator to next tile
    void next() override { ++mfi_; }

    std::unique_ptr<Tile> buildCurrentTile() override;
    Tile*                 buildCurrentTile_forFortran(void) override;

private:
    const unsigned int              level_;     //!< Level of iterator.
    amrex::MultiFab&                unkMFab_;   //!< Ref to multifab for appropriate level.
    std::vector<amrex::MultiFab>&   fluxMFabs_;
    amrex::MFIter                   mfi_;       //!< MFIter that does the iterating.
};

}

#endif

