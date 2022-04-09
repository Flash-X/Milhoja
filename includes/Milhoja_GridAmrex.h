/**
 * \file    Milhoja_GridAmrex.h
 *
 * \brief
 *
 * \todo Does AMReX have a getter for the MPI Communicator so that we don't have
 * this stored in AMReX and here?
 * \todo Can we get the total cells in the domain from AMReX?  If so, then using
 * this and the block size, we can get nBlocks[XYZ].  It think that these values
 * are only used for printing out the config as well.
 * \todo For NDIM < 3, we are storing garbage data in nzb_/nBlocksZ_, etc.  This 
 * seems dangerous and unnecessary.  Change implementation so that we only store
 * values that the library can control.
 */

#ifndef MILHOJA_GRID_AMREX_H__
#define MILHOJA_GRID_AMREX_H__

#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Interpolater.H>

#include "Milhoja.h"
#include "Milhoja_Grid.h"
#include "Milhoja_boundaryConditions.h"
#include "Milhoja_interpolator.h"
#include "Milhoja_actionRoutine.h"

#ifndef MILHOJA_GRID_AMREX
#error "This file need not be compiled if the AMReX backend isn't used"
#endif

namespace milhoja {

/**
  * \brief Derived Grid class for AMReX
  *
  * Grid derived class implemented with AMReX.
  */
class GridAmrex : public Grid,
                  private amrex::AmrCore {
public:
    ~GridAmrex(void);

    GridAmrex(GridAmrex&) = delete;
    GridAmrex(const GridAmrex&) = delete;
    GridAmrex(GridAmrex&&) = delete;
    GridAmrex& operator=(GridAmrex&) = delete;
    GridAmrex& operator=(const GridAmrex&) = delete;
    GridAmrex& operator=(GridAmrex&&) = delete;

    //----- GRID OVERRIDES
    void  finalize(void) override;

    // Pure virtual function overrides.
    void         initDomain(ACTION_ROUTINE initBlock) override;
    void         initDomain(const RuntimeAction& cpuAction) override;
    void         destroyDomain(void) override;
    void         restrictAllLevels(void) override;
    void         fillGuardCells(void) override;
    void         updateGrid(void) override { amrex::AmrCore::regrid(0, 0.0_wp); }
    unsigned int getNGuardcells(void) const override
                    { return static_cast<unsigned int>(nGuard_); }
    unsigned int getNCcVariables(void) const override
                    { return static_cast<unsigned int>(nCcVars_); }
    unsigned int getNFluxVariables(void) const override
                    { return static_cast<unsigned int>(nFluxVars_); }
    void         getBlockSize(unsigned int* nxb,
                              unsigned int* nyb,
                              unsigned int* nzb) const override;
    void         getDomainDecomposition(unsigned int* nBlocksX,
                                        unsigned int* nBlocksY,
                                        unsigned int* nBlocksZ) const override;
    CoordSys     getCoordinateSystem(void) const override;
    IntVect      getDomainLo(const unsigned int lev) const override;
    IntVect      getDomainHi(const unsigned int lev) const override;
    RealVect     getProbLo(void) const override;
    RealVect     getProbHi(void) const override;
    unsigned int getMaxRefinement(void) const override;
    unsigned int getMaxLevel(void) const override;
    unsigned int getNumberLocalBlocks(void) override;
    std::unique_ptr<TileIter> buildTileIter(const unsigned int lev) override;
    TileIter*                 buildTileIter_forFortran(const unsigned int lev) override;
    void         writePlotfile(const std::string& filename,
                               const std::vector<std::string>& names) const override;

    // Other virtual function overrides.
    RealVect     getDeltas(const unsigned int lev) const override;

    //Real        getCellCoord(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& coord) const override;
    Real         getCellFaceAreaLo(const unsigned int axis,
                                   const unsigned int lev,
                                   const IntVect& coord) const override;
    Real         getCellVolume(const unsigned int lev,
                               const IntVect& coord) const override;

    FArray1D     getCellCoords(const unsigned int axis,
                               const unsigned int edge,
                               const unsigned int lev,
                               const IntVect& lo,
                               const IntVect& hi) const override;
    void         fillCellFaceAreasLo(const unsigned int axis,
                                     const unsigned int lev,
                                     const IntVect& lo,
                                     const IntVect& hi,
                                     Real* areaPtr) const override;
    void         fillCellVolumes(const unsigned int lev,
                                 const IntVect& lo,
                                 const IntVect& hi,
                                 Real* volPtr) const override;

private:
    GridAmrex(void);

    // DEV NOTE: needed for polymorphic singleton
    friend Grid& Grid::instance();

    //!< Assume that guardcells are not used when computing fluxes.
    static constexpr   unsigned int NO_GC_FOR_FLUX = 0;

    void    fillPatch(amrex::MultiFab& mf, const int level);

    //----- AMRCORE OVERRIDES
    void MakeNewLevelFromCoarse(int level, amrex::Real time,
                                const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm) override;

    void RemakeLevel(int level,
                     amrex::Real time,
                     const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm) override;

    void ClearLevel(int level) override;

    void MakeNewLevelFromScratch(int level,
                                 amrex::Real time,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override;

    void ErrorEst(int level,
                  amrex::TagBoxArray& tags,
                  amrex::Real time,
                  int ngrow) override;

    //----- STATIC STATE VARIABLES
    static bool    domainInitialized_;
    static bool    domainDestroyed_;

    std::vector<amrex::MultiFab>                unk_;   //!< Physical data, one MF per level
    std::vector<std::vector<amrex::MultiFab>>   fluxes_;  // Flux data
    amrex::Vector<amrex::BCRec>                 bcs_;   //!< Boundary conditions

    // AMReX is given this communicator and therefore should own this.  However,
    // I have not yet found a getter to access it in this class.
    const MPI_Comm        comm_;

    //----- GRID CONFIGURATION VALUES OWNED BY GridAmrex
    // These cannot be obtained from AMReX
    const unsigned int    nBlocksX_, nBlocksY_, nBlocksZ_;
    const unsigned int    nxb_, nyb_, nzb_;
    amrex::Interpolater*  ccInterpolator_;

    // These cannot be acquired from AMReX and play an important role here in
    // terms of constructing MultiFabs.
    //
    // NOTE: nCcVars_ could be retrieved with nComp() from a MultiFab that
    // already exists, but this class has to establish the first MultiFab at
    // some time after construction.
    //
    // We would prefer to store these as unsigned int, but AmrCore works with
    // them as ints.  Therefore, we will eagerly cast and store these results.
    // This implies that code in the class can cast back to unsigned int without
    // first confirming that these variables are non-negative.
    const int   nGuard_;
    const int   nCcVars_;
    const int   nFluxVars_;

    ERROR_ROUTINE   errorEst_;            //!< Routine for marking blocks for refinement
    BC_ROUTINE      extBcFcn_;
    ACTION_ROUTINE  initBlock_noRuntime_; //!< Temporary cache for setting initial conditions without runtime
    RuntimeAction   initCpuAction_;       //!< Temporary cache for setting initial conditions with runtime
};

}

#endif

