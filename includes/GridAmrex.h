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
    void destroyDomain(void) override;

    // Pure virtual function overrides.
    RealVect       getProbLo() const override;
    RealVect       getProbHi() const override;
    void           initDomain(TASK_FCN initBlock) override;
    virtual unsigned int   getMaxRefinement() const override;
    virtual unsigned int   getMaxLevel() const override;
    virtual void     writeToFile(const std::string& filename) const override;

    // Other virtual function overrides.
    virtual RealVect getDeltas(const unsigned int lev) const override;

    //virtual Real     getCellCoord(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& coord) const override;
    virtual Real     getCellFaceAreaLo(const unsigned int axis, const unsigned int lev, const IntVect& coord) const override;
    virtual Real     getCellVolume(const unsigned int lev, const IntVect& coord) const override;

    virtual void     fillCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* coordPtr) const override;
    virtual void     fillCellFaceAreasLo(const unsigned int axis, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* areaPtr) const override;
    virtual void     fillCellVolumes(const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* volPtr) const override;

private:
    friend Grid& Grid::instance();
    GridAmrex(void);

    GridAmrex(const GridAmrex&) = delete;
    GridAmrex(GridAmrex&&) = delete;
    GridAmrex& operator=(const GridAmrex&) = delete;
    GridAmrex& operator=(GridAmrex&&) = delete;
};

}

#endif

