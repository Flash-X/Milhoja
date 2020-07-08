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
#include "Grid_Vector.h"
#include "Grid_RealVect.h"
#include "Grid_IntVect.h"
#include "Tile.h"
#include "runtimeTask.h"

namespace orchestration {

class Grid {
public:
    ~Grid(void);

    static Grid& instance(void);

    void    initDomain(const Real xMin, const Real xMax,
                       const Real yMin, const Real yMax,
                       const Real zMin, const Real zMax,
                       const unsigned int nBlocksX,
                       const unsigned int nBlocksY,
                       const unsigned int nBlocksZ,
                       const unsigned int nVars,
                       TASK_FCN initBlock);
    void    destroyDomain(void);

    amrex::MultiFab&   unk(void)       { return (*unk_); }
    amrex::Geometry&   geometry(void)  { return geometry_; }

    //Basic getter functions.
    Vector<Real>    getDomainLo() const;
    Vector<Real>    getDomainHi() const;
    Vector<Real>    getDeltas(const unsigned int lev) const;
    Vector<Real>    getBlkCenterCoords(const Tile& tileDesc) const;
    Vector<Real>    getCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, const Vector<int> lo, const Vector<int> hi) const;
    Real                 getCellFaceArea(const unsigned int axis, const unsigned int lev, const Vector<int> coord) const;
    Real                 getCellVolume(const unsigned int lev, const Vector<int> coord) const;
    unsigned int               getMaxRefinement() const;
    unsigned int               getMaxLevel() const;

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

