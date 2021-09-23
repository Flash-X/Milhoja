/**
 * \file    Grid.h
 *
 * \brief 
 *
 * It is assumed that MPI has been initialized properly by calling code before
 * the Grid singleton is instantiated.  Calling code must pass in the global
 * communicator that the runtime should use.
 *
 * Once calling code has called the instantiate member function, they are
 * obliged to call finalize *before* finalizing MPI.  This ensures that concrete
 * Grid implementations have the opportunity to perform MPI-based clean-up.
 */

#ifndef GRID_H__
#define GRID_H__

#include <memory>
#include <string>

#include <mpi.h>

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

    static  Grid& instance(void);
    static  void  instantiate(const MPI_Comm comm,
                              const Real xMin, const Real xMax,
                              const Real yMin, const Real yMax,
                              const Real zMin, const Real zMax,
                              const unsigned int nxb,
                              const unsigned int nyb,
                              const unsigned int nzb,
                              const unsigned int nBlocksX,
                              const unsigned int nBlocksY,
                              const unsigned int nBlocksZ,
                              const unsigned int lRefineMax,
                              const unsigned int nGuard,
                              ACTION_ROUTINE initBlock,
                              ERROR_ROUTINE errorEst);
    virtual void  finalize(void);

    // Pure virtual functions that must be implemented by derived class.
    virtual void destroyDomain(void) = 0;
    virtual void initDomain(const unsigned int nDistributorThreads,
                            const unsigned int nRuntimeThreads) = 0;
    virtual void restrictAllLevels() = 0;
    virtual void fillGuardCells() = 0;
    virtual void regrid() = 0;
    virtual IntVect        getDomainLo(const unsigned int lev) const = 0;
    virtual IntVect        getDomainHi(const unsigned int lev) const = 0;
    virtual RealVect       getProbLo() const = 0;
    virtual RealVect       getProbHi() const = 0;
    virtual unsigned int   getMaxRefinement() const = 0;
    virtual unsigned int   getMaxLevel() const = 0;
    virtual unsigned int   getNumberLocalBlocks() = 0;
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
    static bool instantiated_; //!< True if instantiate has been called
    static bool finalized_;    //!< True if finalize has been called 

    static MPI_Comm       globalComm_;

    // These are used just at instantiating to pass configuration data
    // to the concrete Grid implementation so that it can configure
    // itself correctly.  They should not be used otherwise.
    // TODO: Is there a better way to accomplish this?  If not, 
    // prepend cache to each member name?
    static Real           xMin_, xMax_;
    static Real           yMin_, yMax_;
    static Real           zMin_, zMax_;
    static unsigned int   nxb_, nyb_, nzb_;
    static unsigned int   nBlocksX_, nBlocksY_, nBlocksZ_;
    static unsigned int   lRefineMax_;
    static unsigned int   nGuard_;
    static ACTION_ROUTINE initBlock_;
    static ERROR_ROUTINE  errorEst_;
};

}

#endif

