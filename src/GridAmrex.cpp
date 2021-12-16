#include "GridAmrex.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "OrchestrationLogger.h"
#include "GridConfiguration.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "TileIterAmrex.h"

namespace orchestration {

/** Passes FLASH Runtime Parameters to AMReX then initialize AMReX.
  */
GridAmrex::GridAmrex(void)
    : amrcore_{nullptr}
{
    // Check amrex::Real matches orchestraton::Real
    if(!std::is_same<amrex::Real,Real>::value) {
      throw std::logic_error("amrex::Real does not match orchestration::Real");
    }

    // Check IntVect::{I,J,K} behavior matches amrex::Dim3
    IntVect iv{LIST_NDIM(17,19,21)};
    amrex::Dim3 d3 = amrex::IntVect(iv).dim3();
    if( iv.I()!=d3.x || iv.J()!=d3.y || iv.K()!=d3.z ) {
      throw std::logic_error("amrex::Dim3 and orchestration::IntVect do not "
                             "have matching default values.");
    }

    // Access config singleton within limited local scope so that it can't
    // be used for any reason other than configuring AMReX.
    {
        GridConfiguration&   cfg = GridConfiguration::instance();
        if (!cfg.isValid()) {
            throw std::invalid_argument("[GridAmrex::GridAmrex] Invalid configuration");
        }

        amrex::ParmParse    ppGeo("geometry");
        ppGeo.addarr("is_periodic", std::vector<int>{1, 1, 1} );
        ppGeo.add("coord_sys", 0); //cartesian
        ppGeo.addarr("prob_lo", std::vector<Real>{LIST_NDIM(cfg.xMin,
                                                            cfg.yMin,
                                                            cfg.zMin)});
        ppGeo.addarr("prob_hi", std::vector<Real>{LIST_NDIM(cfg.xMax,
                                                            cfg.yMax,
                                                            cfg.zMax)});

        // TODO: Check for overflow
        int  lrefineMax = static_cast<int>(cfg.maxFinestLevel);
        int  nxb        = static_cast<int>(cfg.nxb);
        int  nyb        = static_cast<int>(cfg.nyb);
        int  nzb        = static_cast<int>(cfg.nzb);
        int  nCellsX    = static_cast<int>(cfg.nxb * cfg.nBlocksX);
        int  nCellsY    = static_cast<int>(cfg.nyb * cfg.nBlocksY);
        int  nCellsZ    = static_cast<int>(cfg.nzb * cfg.nBlocksZ);

        amrex::ParmParse ppAmr("amr");

        ppAmr.add("v", 0); //verbosity
        //ppAmr.add("regrid_int",nrefs); //how often to refine
        ppAmr.add("max_level", lrefineMax - 1); //0-based
        ppAmr.addarr("n_cell", std::vector<int>{LIST_NDIM(nCellsX,
                                                          nCellsY,
                                                          nCellsZ)});

        //octree mode:
        ppAmr.add("max_grid_size_x",    nxb);
        ppAmr.add("max_grid_size_y",    nyb);
        ppAmr.add("max_grid_size_z",    nzb);
        ppAmr.add("blocking_factor_x",  nxb * 2);
        ppAmr.add("blocking_factor_y",  nyb * 2);
        ppAmr.add("blocking_factor_z",  nzb * 2);
        ppAmr.add("refine_grid_layout", 0);
        ppAmr.add("grid_eff",           1.0);
        ppAmr.add("n_proper",           1);
        ppAmr.add("n_error_buf",        0);
        ppAmr.addarr("ref_ratio",       std::vector<int>(lrefineMax, 2));

        // Communicate to the config singleton that its contents have been
        // consumed and that other code should not be able to access it.
        cfg.clear();

#ifdef GRID_LOG
        Logger::instance().log("[GridAmrex] Loaded configuration values into AMReX");
#endif
    }

    amrex::Initialize(MPI_COMM_WORLD);

    // Tell Logger to get its rank once AMReX has initialized MPI, but
    // before we log anything
    Logger::instance().acquireRank();
    Logger::instance().log("[GridAmrex] Initialized Grid.");
}

/** Detroy domain and then finalize AMReX.
  */
GridAmrex::~GridAmrex(void) {
    destroyDomain();
    amrex::Finalize();

    Logger::instance().log("[GridAmrex] Finalized Grid.");
}

/** Destroy amrcore_. initDomain can be called again if desired.
  */
void  GridAmrex::destroyDomain(void) {
    if (amrcore_) {
        delete amrcore_; // deletes unk
        amrcore_ = nullptr;
    }

    Logger::instance().log("[GridAmrex] Destroyed domain.");
}

/**
 * initDomain creates the domain in AMReX. It creates amrcore_ and then
 * calls amrex::AmrCore::InitFromScratch.
 *
 * @param initBlock Function pointer to the simulation's initBlock routine.
 * @param nDistributorThreads number of threads to activate in distributor
 * @param nRuntimeThreads     number of threads to use to apply IC
 * @param errorEst            the routine to use estimate errors as part
 *                            of refining blocks
 */
void GridAmrex::initDomain(ACTION_ROUTINE initBlock,
                           const unsigned int nDistributorThreads,
                           const unsigned int nRuntimeThreads,
                           ERROR_ROUTINE errorEst) {
    if (amrcore_) {
        throw std::logic_error("[GridAmrex::initDomain] Grid unit's initDomain"
                               " already called");
    } else if (!initBlock) {
        throw std::logic_error("[GridAmrex::initDomain] Null initBlock function"
                               " pointer given");
    }
    Logger::instance().log("[GridAmrex] Initializing domain...");

    amrcore_ = new AmrCoreFlash(initBlock, nDistributorThreads,
                                nRuntimeThreads, errorEst);
    amrcore_->InitFromScratch(0.0_wp);

    std::string msg = "[GridAmrex] Initialized domain with " +
                      std::to_string(amrcore_->globalNumBlocks()) +
                      " total blocks.";
    Logger::instance().log(msg);
}

void GridAmrex::restrictAllLevels() {
    amrcore_->averageDownAll();
}

/** Fill guard cells on all levels.
  */
void  GridAmrex::fillGuardCells() {
    for(int lev=0; lev<=getMaxLevel(); ++lev) {
#ifdef GRID_LOG
        Logger::instance().log("[GridAmrex] GCFill on level " +
                           std::to_string(lev) );
#endif

        amrex::MultiFab& unk = amrcore_->unk(lev);
        amrcore_->fillPatch(unk, lev);
    }
}


/**
  * getDomainLo gets the lower bound of a given level index space.
  *
  * @return An int vector: <xlo, ylo, zlo>
  */
IntVect    GridAmrex::getDomainLo(const unsigned int lev) const {
    return IntVect{amrcore_->Geom(lev).Domain().smallEnd()};
}

/**
  * getDomainHi gets the upper bound of a given level in index space.
  *
  * @return An int vector: <xhi, yhi, zhi>
  */
IntVect    GridAmrex::getDomainHi(const unsigned int lev) const {
    return IntVect{amrcore_->Geom(lev).Domain().bigEnd()};
}


/**
  * getProbLo gets the physical lower boundary of the domain.
  *
  * @return A real vector: <xlo, ylo, zlo>
  */
RealVect    GridAmrex::getProbLo() const {
    return RealVect{amrcore_->Geom(0).ProbLo()};
}

/**
  * getProbHi gets the physical upper boundary of the domain.
  *
  * @return A real vector: <xhi, yhi, zhi>
  */
RealVect    GridAmrex::getProbHi() const {
    return RealVect{amrcore_->Geom(0).ProbHi()};
}

/**
  * getMaxRefinement returns the maximum possible refinement level which was
  * specified by the user.
  *
  * @return Maximum (finest) refinement level of simulation.
  */
unsigned int GridAmrex::getMaxRefinement() const {
    return amrcore_->maxLevel();
}

/**
  * getMaxLevel returns the highest level of blocks actually in existence.
  *
  * @return The max level of existing blocks (0 is coarsest).
  */
unsigned int GridAmrex::getMaxLevel() const {
    return amrcore_->finestLevel();
}

/**
  * Obtain the total number of blocks managed by the process.
  *
  * \todo Is there an AMReX function to get this information directly?
  *
  * \todo Can we use the AMReX iterator directly here?
  *
  * \todo Make this work for more than one level.
  *
  * \todo Check against FLASH-X to determine if flags are needed.  For
  *       example, would users need # blocks for a given level?  Only count the
  *       number of leaf blocks?
  *
  * \todo This routine should be const.  Right now, the creation of the
  *       iterator does not allow for this.
  *
  * @return The number of local blocks.
  */
unsigned int GridAmrex::getNumberLocalBlocks() {
    unsigned int   nBlocks = 0;
    for (auto ti = buildTileIter(0); ti->isValid(); ti->next()) {
        ++nBlocks;
    }

    return nBlocks;
}

/**
 *
 */
void    GridAmrex::writePlotfile(const std::string& filename,
                                 const std::vector<std::string>& names) const {
    amrex::Vector<std::string>  names_amrex{names.size()};

    for (auto j=0; j<names.size(); ++j) {
        names_amrex[j] = names[j];
    }

    amrcore_->writeMultiPlotfile(filename, names_amrex);
}

/**
  *
  */
std::unique_ptr<TileIter> GridAmrex::buildTileIter(const unsigned int lev) {
    return std::unique_ptr<TileIter>{new TileIterAmrex(amrcore_->unk(lev), lev)};
}


/**
  * getDeltas gets the cell size for a given level.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  */
RealVect    GridAmrex::getDeltas(const unsigned int level) const {
    return RealVect{amrcore_->Geom(level).CellSize()};
}


/** getCellFaceAreaLo gets lo face area of a cell with given integer coordinates
  *
  * @param axis Axis of desired face, returns the area of the lo side.
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return area of face (Real)
  */
Real  GridAmrex::getCellFaceAreaLo(const unsigned int axis,
                                   const unsigned int lev,
                                   const IntVect& coord) const {
    return amrcore_->Geom(lev).AreaLo( amrex::IntVect(coord) , axis);
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  */
Real  GridAmrex::getCellVolume(const unsigned int lev,
                               const IntVect& coord) const {
    return amrcore_->Geom(lev).Volume( amrex::IntVect(coord) );
}

/** Obtain the coordinates along a given axis for either the left edge, center,
  * or right edge of each cell in a given contiguous region of the mesh as
  * specified by lower and upper corners.  Due to the construction of coordinate
  * systems of interest, the coordinate along a single axis only depends on the
  * cell's index for that same axis.  Therefore, the set of coordinates for the
  * region need only be stored as a 1D array.
  *
  * @param axis Axis of desired coord (allowed: Axis::{I,J,K})
  * @param edge Edge of desired coord (allowed: Edge::{Left,Right,Center})
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @returns The coordinates as a Fortran-style array.
  *
  * \todo profile this, see if we can get a version that doesn't require
  * extra copying.
  */
FArray1D    GridAmrex::getCellCoords(const unsigned int axis,
                                     const unsigned int edge,
                                     const unsigned int lev,
                                     const IntVect& lo,
                                     const IntVect& hi) const {
    int   idxLo = 0;
    int   idxHi = 0;
    if        (axis == Axis::I) {
        idxLo = lo.I();
        idxHi = hi.I();
    } else if (axis == Axis::J) {
        idxLo = lo.J();
        idxHi = hi.J();
    } else if (axis == Axis::K) {
        idxLo = lo.K();
        idxHi = hi.K();
    } else {
        throw std::logic_error("GridAmrex::getCellCoords: Invalid axis.");
    }

    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    int nElements = idxHi - idxLo + 1;
    int offset = 0; //accounts for indexing of left/right cases

    FArray1D coords = FArray1D::buildScratchArray1D(idxLo, idxHi);
    if (axis >= NDIM) {
        // TODO: What value to put here?  Should it change
        //       based on edge?
        coords(idxLo) = 0.0;
        return coords;
    }

    //coordvec is length nElements + 1 if edge is Left or Right
    amrex::Vector<amrex::Real> coordvec;
    if        (edge == Edge::Left) {
        amrcore_->Geom(lev).GetEdgeLoc(coordvec,range,axis);
    } else if (edge == Edge::Right) {
        offset = 1;
        amrcore_->Geom(lev).GetEdgeLoc(coordvec,range,axis);
    } else if (edge == Edge::Center) {
        amrcore_->Geom(lev).GetCellLoc(coordvec,range,axis);
    } else {
        throw std::logic_error("GridAmrex::getCellCoords: Invalid edge.");
    }

    //copy results to output
    for(int i=0; i<nElements; ++i) {
        coords(i+idxLo) = coordvec[i+offset];
    }
    return coords;
}

/** fillCellFaceAreasLo fills a Real array (passed by pointer) with the
  * cell face areas in a given range.
  * DEV NOTE: I assumed CoordSys::SetFaceArea corresponds to AreaLo (not AreaHi)
  *
  * @param axis Axis of desired coord (allowed: Axis::{I,J,K})
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param areaPtr Real Ptr to some fortran-style data structure. Will be filled
  *                with areas. Should be of shape:
  *                    (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  */
void    GridAmrex::fillCellFaceAreasLo(const unsigned int axis,
                                       const unsigned int lev,
                                       const IntVect& lo,
                                       const IntVect& hi,
                                       Real* areaPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("GridAmrex::fillCellFaceAreasLo: Invalid axis.");
    }
#endif
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox area_fab{range,1,areaPtr};
    amrcore_->Geom(lev).CoordSys::SetFaceArea(area_fab,range,axis);
}


/** fillCellVolumes fills a Real array (passed by pointer) with the
  * volumes of cells in a given range
  *
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param volPtr Real Ptr to some fortran-style data structure. Will be filled
  *             with volumes. Should be of shape:
  *                 (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  */
void    GridAmrex::fillCellVolumes(const unsigned int lev,
                                   const IntVect& lo,
                                   const IntVect& hi,
                                   Real* volPtr) const {
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox vol_fab{range,1,volPtr};
    amrcore_->Geom(lev).CoordSys::SetVolume(vol_fab,range);
}


}
