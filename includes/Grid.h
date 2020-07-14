/**
 * \file    Grid.h
 *
 * \brief 
 *
 */

#ifndef GRID_H__
#define GRID_H__

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "Grid_REAL.h"
#include "Grid_RealVect.h"
#include "Grid_IntVect.h"
#include "Tile.h"
#include "runtimeTask.h"

namespace orchestration {

class Grid {
public:
    ~Grid(void);

    static Grid& instance(void);

    void    initDomain(const RealVect& probMin,
                       const RealVect& probMax,
                       const IntVect& nBlocks,
                       const unsigned int nVars,
                       TASK_FCN initBlock);
    void    destroyDomain(void);

    amrex::MultiFab&   unk(void)       { return (*unk_); }
    amrex::Geometry&   geometry(void)  { return geometry_; }

    //Basic getter functions.
    RealVect       getDomainLo() const;
    RealVect       getDomainHi() const;
    RealVect       getDeltas(const unsigned int lev) const;
    RealVect       getBlkCenterCoords(const Tile& tileDesc) const;

    Real           getCellCoord(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& coord) const;
    Real           getCellFaceArea(const unsigned int axis, const unsigned int lev, const IntVect& coord) const;
    Real           getCellVolume(const unsigned int lev, const IntVect& coord) const;

    void           fillCellVolumes(const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* volPtr) const;

    unsigned int   getMaxRefinement() const;
    unsigned int   getMaxLevel() const;

    void    writeToFile(const std::string& filename) const;

private:
    Grid(void);
    Grid(const Grid&) = delete;
    Grid(const Grid&&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid& operator=(const Grid&&) = delete;

    amrex::Geometry    geometry_;
    amrex::MultiFab*   unk_;
};

}

#endif

