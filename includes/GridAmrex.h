/**
 * \file    GridAmrex.h
 *
 * \brief
 *
 */

#ifndef GRIDAMREX_H__
#define GRIDAMREX_H__

#include "Grid.h"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include "Grid_AmrCoreFlash.h"

namespace orchestration {

class GridAmrex : public Grid {
public:
    ~GridAmrex(void);

    GridAmrex(GridAmrex&) = delete;
    GridAmrex(const GridAmrex&) = delete;
    GridAmrex(GridAmrex&&) = delete;
    GridAmrex& operator=(GridAmrex&) = delete;
    GridAmrex& operator=(const GridAmrex&) = delete;
    GridAmrex& operator=(GridAmrex&&) = delete;

    // Pure virtual function overrides.
    void         initDomain(ACTION_ROUTINE initBlock) override;
    void         destroyDomain(void) override;
    IntVect      getDomainLo(const unsigned int lev) const override;
    IntVect      getDomainHi(const unsigned int lev) const override;
    RealVect     getProbLo() const override;
    RealVect     getProbHi() const override;
    unsigned int getMaxRefinement() const override;
    unsigned int getMaxLevel() const override;
    void         writeToFile(const std::string& filename) const override;
    std::unique_ptr<TileIter> buildTileIter(const unsigned int lev) override;

    // Other virtual function overrides.
    RealVect     getDeltas(const unsigned int lev) const override;

    //Real        getCellCoord(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& coord) const override;
    Real         getCellFaceAreaLo(const unsigned int axis,
                                   const unsigned int lev,
                                   const IntVect& coord) const override;
    Real         getCellVolume(const unsigned int lev,
                               const IntVect& coord) const override;

    FArray1D     getCellCoords(const unsigned int axis,
                               const unsigned int edge,
                               const unsigned int lev,
                               const IntVect& lo,
                               const IntVect& hi) const override;
    void         fillCellFaceAreasLo(const unsigned int axis,
                                     const unsigned int lev,
                                     const IntVect& lo,
                                     const IntVect& hi,
                                     Real* areaPtr) const override;
    void         fillCellVolumes(const unsigned int lev,
                                 const IntVect& lo,
                                 const IntVect& hi,
                                 Real* volPtr) const override;

private:
    GridAmrex(void);

    // DEV NOTE: needed for polymorphic singleton
    friend Grid& Grid::instance();

    // DEV NOTE: Used mix-in pattern over inheritance so amrcore can be
    // created/destroyed multiple times in one run.
    AmrCoreFlash*      amrcore_;
};

}

#endif

