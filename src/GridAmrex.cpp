#include "GridAmrex.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include "TileIterAmrex.h"

#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "OrchestrationLogger.h"

#include "Flash.h"
#include "constants.h"

namespace orchestration {

void passRPToAmrex() {
#ifdef GRID_LOG
    Logger::instance().log("[GridAmrex] Passing runtime parameters"
                           " to amrex...");
#endif
    {
        amrex::ParmParse pp("geometry");
        pp.addarr("is_periodic", std::vector<int>{1,1,1} );
        pp.add("coord_sys",0); //cartesian
        pp.addarr("prob_lo",std::vector<Real>{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)});
        pp.addarr("prob_hi",std::vector<Real>{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)});
    }

    {
        amrex::ParmParse pp("amr");

        pp.add("v",0); //verbosity
        //pp.add("regrid_int",nrefs); //how often to refine
        pp.add("max_level",LREFINE_MAX-1); //0-based
        pp.addarr("n_cell",std::vector<int>{LIST_NDIM(NXB*N_BLOCKS_X,
                                      NYB*N_BLOCKS_Y,
                                      NZB*N_BLOCKS_Z)});

        //octree mode:
        pp.add("max_grid_size_x",NXB);
        pp.add("max_grid_size_y",NYB);
        pp.add("max_grid_size_z",NZB);
        pp.add("blocking_factor_x",NXB*2);
        pp.add("blocking_factor_y",NYB*2);
        pp.add("blocking_factor_z",NZB*2);
        pp.add("refine_grid_layout",0);
        pp.add("grid_eff",1.0);
        pp.add("n_proper",1);
        pp.add("n_error_buf",0);
        pp.addarr("ref_ratio",std::vector<int>(LREFINE_MAX,2));
    }

}

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

    passRPToAmrex();
    amrex::Initialize(MPI_COMM_WORLD);

#ifdef GRID_LOG
    Logger::instance().log("[GridAmrex] Initialized Grid.");
#endif
}

/** Detroy domain and then finalize AMReX.
  */
GridAmrex::~GridAmrex(void) {
    destroyDomain();
    amrex::Finalize();

#ifdef GRID_LOG
    Logger::instance().log("[GridAmrex] Finalized Grid.");
#endif
}

/** Destroy amrcore_. initDomain can be called again if desired.
  */
void  GridAmrex::destroyDomain(void) {
    if (amrcore_) {
        delete amrcore_; // deletes unk
        amrcore_ = nullptr;
    }
#ifdef GRID_LOG
    Logger::instance().log("[GridAmrex] Destroyed domain.");
#endif
}

/**
 * initDomain creates the domain in AMReX. It creates amrcore_ and then
 * calls amrex::AmrCore::InitFromScratch.
 *
 * @param initBlock Function pointer to the simulation's initBlock routine.
 */
void GridAmrex::initDomain(ACTION_ROUTINE initBlock,
                           ERROR_ROUTINE errorEst) {
    if (amrcore_) {
        throw std::logic_error("[GridAmrex::initDomain] Grid unit's initDomain"
                               " already called");
    } else if (!initBlock) {
        throw std::logic_error("[GridAmrex::initDomain] Null initBlock function"
                               " pointer given");
    }
#ifdef GRID_LOG
    Logger::instance().log("[GridAmrex] Initializing domain...");
#endif

    amrcore_ = new AmrCoreFlash(initBlock,errorEst);
    amrcore_->InitFromScratch(0.0_wp);

    amrcore_->averageDownAll();

#ifdef GRID_LOG
    std::string msg = "[GridAmrex] Initialized domain with " +
                      std::to_string(amrcore_->globalNumBlocks()) +
                      " total blocks.";
    Logger::instance().log(msg);
#endif
}

/** Fill guard cells on a level.
  */
void  GridAmrex::fillGC(const unsigned int lev) {
#ifdef GRID_LOG
    Logger::instance().log("[GridAmrex] GCFill on level " +
                           std::to_string(lev) );
#endif
    amrcore_->fillPatch(amrcore_->unk(lev),lev);
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
 *
 */
void    GridAmrex::writePlotfile(const std::string& filename) const {
    amrcore_->writeMultiPlotfile(filename);
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

/** fillCellCoords fills a Real array (passed by pointer) with the
  * cell coordinates in a given range
  *
  * @param axis Axis of desired coord (allowed: Axis::{I,J,K})
  * @param edge Edge of desired coord (allowed: Edge::{Left,Right,Center})
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param coordPtr Real Ptr to array of length hi[axis]-lo[axis]+1.
  *
  * \todo profile this, see if we can get a version that doesn't require
  * extra copying.
  */
void    GridAmrex::fillCellCoords(const unsigned int axis,
                                  const unsigned int edge,
                                  const unsigned int lev,
                                  const IntVect& lo,
                                  const IntVect& hi, Real* coordPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("GridAmrex::fillCellCoords: Invalid axis.");
    }
    if(edge!=Edge::Left && edge!=Edge::Right && edge!=Edge::Center){
        throw std::logic_error("GridAmrex::fillCellCoords: Invalid edge.");
    }
#endif
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    int nElements = hi[axis] - lo[axis] + 1;
    int offset = 0; //accounts for indexing of left/right cases

    //coordvec is length nElements + 1 if edge is Left or Right
    amrex::Vector<amrex::Real> coordvec;
    switch (edge) {
        case Edge::Left:
            amrcore_->Geom(lev).GetEdgeLoc(coordvec,range,axis);
            break;
        case Edge::Right:
            offset = 1;
            amrcore_->Geom(lev).GetEdgeLoc(coordvec,range,axis);
            break;
        case Edge::Center:
            amrcore_->Geom(lev).GetCellLoc(coordvec,range,axis);
            break;
    }

    //copy results to output
    for(int i=0; i<nElements; ++i) {
        coordPtr[i] = coordvec[i+offset];
    }
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
  * @param vols Real Ptr to some fortran-style data structure. Will be filled
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
