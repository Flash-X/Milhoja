/**
 * \file    Milhoja_Grid.h
 *
 * \brief 
 *
 */

#ifndef MILHOJA_GRID_H__
#define MILHOJA_GRID_H__

#include <memory>
#include <string>
#include <vector>

#include "Milhoja_real.h"
#include "Milhoja_RealVect.h"
#include "Milhoja_IntVect.h"
#include "Milhoja_coordinateSystem.h"
#include "Milhoja_FArray1D.h"
#include "Milhoja_Tile.h"
#include "Milhoja_TileIter.h"
#include "Milhoja_TileWrapper.h"
#include "Milhoja_actionRoutine.h"
#include "Milhoja_RuntimeAction.h"

namespace milhoja {

using INIT_BLOCK_NO_RUNTIME = void (*)(Tile* tileDesc);

/**
  * Grid is an abstract base class designed with the singleton pattern.
  * Each AMR package will have a corresponding derived class from Grid,
  * implementing at least the pure virtual functions.
  *
  * \todo Now that initDomain/destroyDomain can only be called once, is
  * destroyDomain necessary?  Why not just move that code to finalize?
  */
class Grid {
public:
    virtual ~Grid(void);

    Grid(Grid&) = delete;
    Grid(const Grid&) = delete;
    Grid(Grid&&) = delete;
    Grid& operator=(Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid& operator=(Grid&&) = delete;

    static  Grid& instance(void);
    static  void  initialize(void);
    virtual void  finalize(void);

    // Pure virtual functions that must be implemented by derived class.
#ifdef FULL_MILHOJAGRID
    virtual void destroyDomain(void) = 0;
    virtual void initDomain(INIT_BLOCK_NO_RUNTIME initBlock) = 0;
    virtual void initDomain(const RuntimeAction& cpuAction,
                            const TileWrapper* prototype) = 0;
    virtual void restrictAllLevels() = 0;
    virtual void fillGuardCells() = 0;
    virtual void updateGrid() = 0;
    virtual unsigned int   getNGuardcells(void) const = 0;
    virtual unsigned int   getNCcVariables(void) const = 0;
    virtual unsigned int   getNFluxVariables(void) const = 0;
    virtual void           getBlockSize(unsigned int* nxb,
                                        unsigned int* nyb,
                                        unsigned int* nzb) const = 0;
    virtual void           getDomainDecomposition(unsigned int* nBlocksX,
                                                  unsigned int* nBlocksY,
                                                  unsigned int* nBlocksZ) const = 0;
    virtual CoordSys       getCoordinateSystem(void) const = 0;
    virtual IntVect        getDomainLo(const unsigned int lev) const = 0;
    virtual IntVect        getDomainHi(const unsigned int lev) const = 0;
    virtual RealVect       getProbLo() const = 0;
    virtual RealVect       getProbHi() const = 0;
    virtual unsigned int   getMaxRefinement() const = 0;
    virtual unsigned int   getNumberLocalBlocks() = 0;
#endif
    virtual unsigned int   getMaxLevel(void) const = 0;
    virtual std::unique_ptr<TileIter> buildTileIter(const unsigned int lev) = 0;
#ifdef FULL_MILHOJAGRID
    virtual TileIter*                 buildTileIter_forFortran(const unsigned int lev) = 0;
    virtual void writePlotfile(const std::string& filename,
                               const std::vector<std::string>& names) const = 0;
#endif

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

    static bool initialized_;  //!< True if initialize has been called
    static bool finalized_;    //!< True if finalize has been called
};

}

#endif

