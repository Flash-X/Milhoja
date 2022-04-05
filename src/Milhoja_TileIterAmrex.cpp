#include "Milhoja_TileIterAmrex.h"

//#include <iostream>

#include "Milhoja_TileAmrex.h"

namespace milhoja {

/**
 * Construct an AMReX tile iterator for iterating over a subset of the tiles in
 * the given level.
 *
 * \param unk     The unk data structure at the given level
 * \param fluxes  The collection of flux data structures at the given level for
 *                all MILHOJA_NDIM dimensions in the problem.  If the problem
 *                does not have any flux variables, then the given vector should
 *                be empty.
 * \param level   The AMR level whose tiles are to be iterated over
 */
TileIterAmrex::TileIterAmrex(amrex::MultiFab& unk,
                             std::vector<amrex::MultiFab>& fluxes,
                             const unsigned int level)
    : level_{level},
      unkMFab_{unk},
      fluxMFabs_{fluxes},
      mfi_{unkMFab_}
{
//    std::cout << "Number flux MFabs = " << fluxMFabs_.size() << std::endl;
}

TileIterAmrex::~TileIterAmrex() {
}

/**
 * Obtain the Tile object currently indexed by the iterator.  It is intended
 * that all C++ code use this version of buildCurrentTile.
 *
 * \todo Study how to implement tiling properly here.  Fix in forFortran version as
 * well.
 *
 * \returns The Tile object.
 */
std::unique_ptr<Tile>   TileIterAmrex::buildCurrentTile(void) {
    int  gridIdx{mfi_.index()};
    int  tileIdx{mfi_.LocalTileIndex()};

    // We'd prefer to get references to the FABs here, but we cannot store
    // references in standard containers since the elements of containers must
    // be alterable.  Therefore, we are forced to get pointers.  However, if
    // calling code follows the use rules of Tiles, then these pointers will
    // never be dangling and the danger is less.
    std::vector<amrex::FArrayBox*>    fluxPtrs{fluxMFabs_.size(), nullptr};
    for (auto i=0; i<fluxMFabs_.size(); ++i) {
        fluxPtrs[i] = &(fluxMFabs_[i][gridIdx]);
//        std::cout << "Flux pointer " << i << " for grid " << gridIdx 
//                  << " on level " << level_ << std::endl;
    }

    return std::unique_ptr<Tile>{ new TileAmrex(level_, gridIdx, tileIdx,
                                                mfi_.tilebox(),
                                                mfi_.fabbox(),
                                                unkMFab_[gridIdx],
                                                std::move(fluxPtrs)) };
}

/** 
 * Obtain the Tile object currently indexed by the iterator.  It is intended
 * that this version only be used in the Fortran/C++ interoperability interface.
 *
 * Calling code owns the object and is responsible for "delete"ing it.
 *
 * \returns The Tile object.
 */
Tile*   TileIterAmrex::buildCurrentTile_forFortran(void) {
    int  gridIdx{mfi_.index()};
    int  tileIdx{mfi_.LocalTileIndex()};

    // See comments in same location in buildCurrentTile().
    std::vector<amrex::FArrayBox*>    fluxPtrs{fluxMFabs_.size(), nullptr};
    for (auto i=0; i<fluxMFabs_.size(); ++i) {
//        std::cout << "Flux pointer " << i << " for grid " << gridIdx 
//                  << " on level " << level_ << std::endl;
        fluxPtrs[i] = &(fluxMFabs_[i][gridIdx]);
    }

    return (new TileAmrex(level_, gridIdx, tileIdx,
                          mfi_.tilebox(),
                          mfi_.fabbox(),
                          unkMFab_[gridIdx],
                          std::move(fluxPtrs)) );
}

}

