/**
 * \file    Grid.h
 *
 * \brief 
 *
 */

#ifndef GRID_H__
#define GRID_H__

//TODO delete these
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include "Grid_AmrCoreFlash.h"

#include "Grid_REAL.h"
#include "Grid_RealVect.h"
#include "Grid_IntVect.h"
#include "Tile.h"
#include "TileIter.h"
#include "runtimeTask.h"

namespace orchestration {

class Grid {
public:
    ~Grid(void) { instantiated_ = false; }

    static Grid& instance(void);
    static void  instantiate(void);
    virtual void destroyDomain(void) {}

    amrex::MultiFab&   unk(void)       { return (*unk_); }
    amrex::Geometry&   geometry(void)  { return amrcore_->Geom(0); }

    // Pure virtual functions that must be implemented by derived class.
    virtual void  initDomain(TASK_FCN initBlock) = 0;
    virtual RealVect       getProbLo() const = 0;
    virtual RealVect       getProbHi() const = 0;
    virtual unsigned int   getMaxRefinement() const = 0;
    virtual unsigned int   getMaxLevel() const = 0;
    virtual void     writeToFile(const std::string& filename) const = 0;
    virtual TileIter buildTileIter(const unsigned int lev) = 0;


    // Virtual functions with a default implementation that may be
    // overwritten by derived class.
    virtual RealVect getDeltas(const unsigned int lev) const;
    virtual RealVect getBlkCenterCoords(const Tile& tileDesc) const;

    //virtual Real     getCellCoord(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& coord) const;
    virtual Real     getCellFaceAreaLo(const unsigned int axis, const unsigned int lev, const IntVect& coord) const;
    virtual Real     getCellVolume(const unsigned int lev, const IntVect& coord) const;

    virtual void     fillCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* coordPtr) const;
    virtual void     fillCellFaceAreasLo(const unsigned int axis, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* areaPtr) const;
    virtual void     fillCellVolumes(const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* volPtr) const;

    amrex::MultiFab*   unk_;

protected:
    Grid(void) : unk_(nullptr) {}
    static bool instantiated_;
    AmrCoreFlash*      amrcore_;

// TODO make these private??
    Grid(const Grid&) = delete;
    Grid(Grid&&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid& operator=(Grid&&) = delete;
};

}

#endif

