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

#include "Grid_REAL.h"
#include "Grid_RealVect.h"
#include "Grid_IntVect.h"
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
                            ERROR_ROUTINE errorEst) = 0;
    virtual IntVect        getDomainLo(const unsigned int lev) const = 0;
    virtual IntVect        getDomainHi(const unsigned int lev) const = 0;
    virtual RealVect       getProbLo() const = 0;
    virtual RealVect       getProbHi() const = 0;
    virtual unsigned int   getMaxRefinement() const = 0;
    virtual unsigned int   getMaxLevel() const = 0;
    virtual std::unique_ptr<TileIter> buildTileIter(const unsigned int lev) = 0;
    virtual void writePlotfile(const std::string& filename) const = 0;


    // Virtual functions with a default implementation that may be
    // overwritten by derived class.
    virtual RealVect getDeltas(const unsigned int lev) const;

    //virtual Real     getCellCoord(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& coord) const;
    virtual Real     getCellFaceAreaLo(const unsigned int axis,
                                       const unsigned int lev,
                                       const IntVect& coord) const;
    virtual Real     getCellVolume(const unsigned int lev,
                                   const IntVect& coord) const;

    virtual void     fillCellCoords(const unsigned int axis,
                                    const unsigned int edge,
                                    const unsigned int lev,
                                    const IntVect& lo,
                                    const IntVect& hi,
                                    Real* coordPtr) const;
    virtual void     fillCellFaceAreasLo(const unsigned int axis,
                                         const unsigned int lev,
                                         const IntVect& lo,
                                         const IntVect& hi,
                                         Real* areaPtr) const;
    virtual void     fillCellVolumes(const unsigned int lev,
                                     const IntVect& lo,
                                     const IntVect& hi,
                                     Real* volPtr) const;

protected:
    Grid(void) {}
    static bool instantiated_; //!< Track if singleton has been instantiated.

};

}

#endif

