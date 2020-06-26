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

#include "Tile.h"
#include "runtimeTask.h"

class Grid {
public:
    ~Grid(void);

    static Grid& instance(void);

    void    initDomain(const amrex::Real xMin, const amrex::Real xMax,
                       const amrex::Real yMin, const amrex::Real yMax,
                       const amrex::Real zMin, const amrex::Real zMax,
                       const unsigned int nBlocksX,
                       const unsigned int nBlocksY,
                       const unsigned int nBlocksZ,
                       const unsigned int nVars,
                       TASK_FCN initBlock);
    void    destroyDomain(void);

    amrex::MultiFab&   unk(void)       { return (*unk_); }
    amrex::Geometry&   geometry(void)  { return geometry_; }

    std::vector<double>    getDomainLo();
    std::vector<double>    getDomainHi();
    std::vector<double>    getDeltas(const unsigned int lev);
    std::vector<double>    getBlkCenterCoords(Tile blockDesc);
    std::vector<double>    getCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, std::vector<int> lo, std::vector<int> hi);
    double                 getCellFaceArea(const unsigned int axis, const unsigned int lev, std::vector<int> coord);
    double                 getCellVolume(const unsigned int lev, std::vector<int> coord);
    unsigned int           getMaxRefinement();
    unsigned int           getMaxLevel();

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

