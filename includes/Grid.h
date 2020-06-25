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

    void    getDomainBoundBox(double bbox[][3]);
    void    getDeltas(int lev, double deltas[]);
    void    getBlkCenterCoords(Tile blockDesc, double blockCenter[]);
    void    getCellCoords(int axis, int edge, int lev, int lo[], int hi[], double coord[]);
    void    getCellFaceAreas(int axis, int lev, int lo[], int hi[], double areas[][3]);
    void    getCellVolumes(int lev, int lo[], int hi[], int volumes[][3]);
    void    getMaxRefinement(int* maxRef, int mode);
    void    updateRefinement(int* nstep, double* time, bool* gridChanged);

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

