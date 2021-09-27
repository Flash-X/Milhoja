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
 * \brief Fully-implemented Tile class for use with AMReX.
 *
 * As part of simplifying the implementation and debugging of the Tile Fortran/C
 * interoperability layer, I have decided to cache all tile metadata at
 * instantiation.
 *
 * This was motivated just as a temporary development step and is not a final
 * design decision.  It would have to be determined through performance testing
 * with both pure C++ simulations and with Flash-X if this increased use of
 * memory resources is offset by significant performance increases.
 */
class TileAmrex
    : public Tile
{
public:
    TileAmrex(amrex::MFIter& itor, amrex::MultiFab& unkRef,
              const unsigned int level);
    ~TileAmrex(void);

    TileAmrex(TileAmrex&&) = delete;
    TileAmrex& operator=(TileAmrex&&) = delete;
    TileAmrex(const TileAmrex&) = delete;
    TileAmrex(TileAmrex&) = delete;
    TileAmrex& operator=(TileAmrex&) = delete;
    TileAmrex& operator=(const TileAmrex&) = delete;

    // Overrides to pure virtual functions
    bool           isNull(void) const override;
    int            gridIndex(void) const override { return gridIdx_; }
    unsigned int   level(void) const override { return level_; }

    IntVect        lo(void) const override;
    IntVect        hi(void) const override;
    IntVect        loGC(void) const override;
    IntVect        hiGC(void) const override;
    void           lo(int* i, int* j, int* k) const override;
    void           hi(int* i, int* j, int* k) const override;
    void           loGC(int* i, int* j, int* k) const override;
    void           hiGC(int* i, int* j, int* k) const override;

    FArray4D       data(void) override;
    Real*          dataPtr(void) override;

private:
    unsigned int  level_;    /**< 0-based level of Tile */
    int           gridIdx_;  /**< Index into multifab */
    unsigned int  nCcVars_;  /**< Number of variables in UNK */
    Real*         dataPtr_;
    int           lo_[MDIM];
    int           hi_[MDIM];
    int           loGC_[MDIM];
    int           hiGC_[MDIM];
};


}
#endif

