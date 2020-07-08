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
#include "Tile.h"
#include "runtimeTask.h"

class Grid {
public:
    ~Grid(void);

    static Grid& instance(void);

    void    initDomain(const grid::Real xMin, const grid::Real xMax,
                       const grid::Real yMin, const grid::Real yMax,
                       const grid::Real zMin, const grid::Real zMax,
                       const unsigned int nBlocksX,
                       const unsigned int nBlocksY,
                       const unsigned int nBlocksZ,
                       const unsigned int nVars,
                       TASK_FCN initBlock);
    void    destroyDomain(void);

    amrex::MultiFab&   unk(void)       { return (*unk_); }
    amrex::Geometry&   geometry(void)  { return geometry_; }

    //Basic getter functions.
    grid::Vector<grid::Real>    getDomainLo();
    grid::Vector<grid::Real>    getDomainHi();
    grid::Vector<grid::Real>    getDeltas(const unsigned int lev);
    grid::Vector<grid::Real>    getBlkCenterCoords(const Tile& tileDesc);
    grid::Vector<grid::Real>    getCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, const grid::Vector<int> lo, const grid::Vector<int> hi);
    grid::Real                 getCellFaceArea(const unsigned int axis, const unsigned int lev, const grid::Vector<int> coord);
    grid::Real                 getCellVolume(const unsigned int lev, const grid::Vector<int> coord);
    unsigned int               getMaxRefinement();
    unsigned int               getMaxLevel();

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

#endif

