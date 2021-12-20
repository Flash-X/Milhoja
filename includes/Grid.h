/**
 * \file    Grid.h
 *
 * \brief 
 *
 */

#ifndef GRID_H__
#define GRID_H__

#include <memory>
#include <string>
#include <vector>

#include "Grid_REAL.h"
#include "Grid_RealVect.h"
#include "Grid_IntVect.h"
#include "FArray1D.h"
#include "Tile.h"
#include "TileIter.h"
#include "actionRoutine.h"

namespace orchestration {

/**
  * Grid is an abstract base class designed with the singleton pattern.
  * Each AMR package will have a corresponding derived class from Grid,
  * implementing at least the pure virtual functions.
  */
class Grid {
public:
    virtual ~Grid(void) { instantiated_ = false; }

    Grid(Grid&) = delete;
    Grid(const Grid&) = delete;
    Grid(Grid&&) = delete;
    Grid& operator=(Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid& operator=(Grid&&) = delete;

    static Grid& instance(void);
    static void  instantiate(void);

    // Pure virtual functions that must be implemented by derived class.
    virtual void destroyDomain(void) = 0;
    virtual void initDomain(ACTION_ROUTINE initBlock,
                            const unsigned int nDistributorThreads,
                            const unsigned int nRuntimeThreads,
                            ERROR_ROUTINE errorEst) = 0;
    virtual void restrictAllLevels() = 0;
    virtual void fillGuardCells() = 0;
    virtual void regrid() = 0;
    virtual void           getBlockSize(unsigned int* nxb,
                                        unsigned int* nyb,
                                        unsigned int* nzb) const = 0;
    virtual IntVect        getDomainLo(const unsigned int lev) const = 0;
    virtual IntVect        getDomainHi(const unsigned int lev) const = 0;
    virtual RealVect       getProbLo() const = 0;
    virtual RealVect       getProbHi() const = 0;
    virtual unsigned int   getMaxRefinement() const = 0;
    virtual unsigned int   getMaxLevel() const = 0;
    virtual unsigned int   getNumberLocalBlocks() = 0;
    virtual std::unique_ptr<TileIter> buildTileIter(const unsigned int lev) = 0;
    virtual void writePlotfile(const std::string& filename,
                               const std::vector<std::string>& names) const = 0;


    // Virtual functions with a default implementation that may be
    // overwritten by derived class.
    virtual RealVect getDeltas(const unsigned int lev) const;

    //virtual Real     getCellCoord(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& coord) const;
    virtual Real     getCellFaceAreaLo(const unsigned int axis,
                                       const unsigned int lev,
                                       const IntVect& coord) const;
    virtual Real     getCellVolume(const unsigned int lev,
                                   const IntVect& coord) const;

    virtual FArray1D getCellCoords(const unsigned int axis,
                                   const unsigned int edge,
                                   const unsigned int lev,
                                   const IntVect& lo,
                                   const IntVect& hi) const;
    virtual void     fillCellFaceAreasLo(const unsigned int axis,
                                         const unsigned int lev,
                                         const IntVect& lo,
                                         const IntVect& hi,
                                         Real* areaPtr) const;
    virtual void     fillCellVolumes(const unsigned int lev,
                                     const IntVect& lo,
                                     const IntVect& hi,
                                     Real* volPtr) const;

    virtual void     subcellGeometry(const unsigned int nsubI,
                                     const unsigned int nsubJ,
                                     const unsigned int nsubK,
                                     const Real dvCell,
                                     Real* dvSubPtr,
                                     const Real xL = 0.0_wp,
                                     const Real xR = 0.0_wp,
                                     const Real yL = 0.0_wp,
                                     const Real yR = 0.0_wp);

protected:
    Grid(void) {}
    static bool instantiated_; //!< Track if singleton has been instantiated.

};

}

#endif

