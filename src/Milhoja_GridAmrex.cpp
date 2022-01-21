#include "Milhoja_GridAmrex.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_FillPatchUtil.H>

#include "Milhoja_Logger.h"
#include "Milhoja_GridConfiguration.h"
#include "Milhoja_axis.h"
#include "Milhoja_edge.h"
#include "Milhoja_TileIterAmrex.h"
#include "Milhoja_RuntimeAction.h"
#include "Milhoja_ThreadTeamDataType.h"
#include "Milhoja_ThreadTeam.h"

namespace milhoja {

/**
  * Construct the AMReX Grid backend singleton.  When construction ends,
  * both AMReX and MPI are fully initialized.  However, the data structures
  * needed to store data will not have been created.
  *
  * It is assumed that the GridConfiguration singleton has already been loaded,
  * which implies that rudimentary validation of configuration values has been
  * carried out.  Therefore, only minimal error checking is done here.
  */
GridAmrex::GridAmrex(void)
    : Grid(),
      AmrCore(),
      nxb_{GridConfiguration::instance().nxb}, 
      nyb_{GridConfiguration::instance().nyb}, 
      nzb_{GridConfiguration::instance().nzb},
      nGuard_{static_cast<int>(GridConfiguration::instance().nGuard)},
      nCcVars_{static_cast<int>(GridConfiguration::instance().nCcVars)},
      initBlock_{GridConfiguration::instance().initBlock},
      nThreads_initBlock_{GridConfiguration::instance().nCpuThreads_init},
      nDistributorThreads_initBlock_{GridConfiguration::instance().nDistributorThreads_init},
      errorEst_{GridConfiguration::instance().errorEstimation}
{
#ifndef USE_THREADED_DISTRIBUTOR
      // Override if multithreading is disabled
      nDistributorThreads_initBlock_ = 1;
#endif

    // Satisfy grid configuration requirements and suggestions (See dev guide).
    {
        GridConfiguration&  cfg = GridConfiguration::instance();

        // TODO: Retroactively check cast to int of cfg.nGuard and cfg.nCcVars
        // for overflow and fail if the stored values are invalid.  We are
        // forced to cast and then check since we want nGuard_/nCcVars_ to be
        // const.

        cfg.clear();
    }

    // Check amrex::Real matches orchestraton::Real
    if(!std::is_same<amrex::Real,Real>::value) {
      throw std::logic_error("amrex::Real does not match milhoja::Real");
    }

    // Check IntVect::{I,J,K} behavior matches amrex::Dim3
    IntVect iv{LIST_NDIM(17,19,21)};
    amrex::Dim3 d3 = amrex::IntVect(iv).dim3();
    if( iv.I()!=d3.x || iv.J()!=d3.y || iv.K()!=d3.z ) {
      throw std::logic_error("amrex::Dim3 and milhoja::IntVect do not "
                             "have matching default values.");
    }

    // Allocate and resize unk_
    unk_.resize(max_level+1);

    // Set periodic Boundary conditions
    bcs_.resize(1);
    for(int i=0; i<MILHOJA_NDIM; ++i) {
        bcs_[0].setLo(i, amrex::BCType::int_dir);
        bcs_[0].setHi(i, amrex::BCType::int_dir);
    }

    Logger::instance().log("[GridAmrex] Initialized Grid.");
}

/**
  * Clean-up and finalize the AMReX Grid singleton.  This includes finalizing
  * AMReX.
  *
  * Under normal program execution and if initDomain has been called, it is a
  * logical error for singleton destruction to occur without the calling code
  * having first called destroyDomain.
  */
GridAmrex::~GridAmrex(void) {
    destroyDomain();

    // AmrCore will finalize only after this destructor runs and the compiler
    // determines when that will happen.  Therefore, we must finalize AMReX here
    // and need to clean-up manually before doing so.  Else, the automatic
    // clean-up occurs after AMReX is finalized, which results in runtime
    // failures.
    std::vector<amrex::MultiFab>().swap(unk_);

    amrex::Finalize();

    Logger::instance().log("[GridAmrex] Finalized Grid.");
}

/**
 *  Destroy the domain.  It is a logical error to call this if initDomain has
 *  not already been called or to call it multiple times.
 *
 * @todo  This routine shall throw an error if it is called more than once.
 */
void  GridAmrex::destroyDomain(void) {
    initBlock_                     = nullptr;
    nThreads_initBlock_            = 0;
    nDistributorThreads_initBlock_ = 0;
    errorEst_                      = nullptr;
    Logger::instance().log("[GridAmrex] Destroyed domain.");
}

/**
 * Set the initial conditions and setup the grid structure so that the initial
 * conditions are resolved in accord with the Grid configuration.
 *
 * @todo  This routine shall throw an error if it is called more than once.
 */
void GridAmrex::initDomain(void) {
    Logger::instance().log("[GridAmrex] Initializing domain...");

    InitFromScratch(0.0_wp);

    std::vector<amrex::MultiFab>::size_type   nGlobalBlocks = 0;
    for(unsigned int level=0; level<=finest_level; ++level) {
        nGlobalBlocks += unk_[level].size();
    }

    std::string msg = "[GridAmrex] Initialized domain with " +
                      std::to_string(nGlobalBlocks) +
                      " total blocks.";
    Logger::instance().log(msg);
}

/**
 * Where feasible, set the data in the cells of all coarse blocks to the data
 * values obtained by aggregating the data from the associated cells from finer
 * blocks.
 *
 * @todo Rename so that the action doesn't refer to the action carried out by
 * the algorithm, but rather reflects what action needs to be carried out on the
 * data.
 */
void GridAmrex::restrictAllLevels() {
    for (int level=finest_level-1; level>=0; --level) {
        amrex::average_down(unk_[level+1], unk_[level],
                            geom[level+1], geom[level],
                            0, nCcVars_, ref_ratio[level]);
    }
}

/** Fill guard cells on all levels.
  */
void  GridAmrex::fillGuardCells() {
    for (int level=0; level<=getMaxLevel(); ++level) {
#ifdef GRID_LOG
        Logger::instance().log("[GridAmrex] GCFill on level " +
                           std::to_string(level) );
#endif

        fillPatch(unk_[level], level);
    }
}

/**
 * Obtain the size of the interior of all blocks.
 */
void    GridAmrex::getBlockSize(unsigned int* nxb,
                                unsigned int* nyb,
                                unsigned int* nzb) const {
    *nxb = nxb_;
    *nyb = nyb_;
    *nzb = nzb_;
}

/**
  * getDomainLo gets the lower bound of a given level index space.
  *
  * @return An int vector: <xlo, ylo, zlo>
  */
IntVect    GridAmrex::getDomainLo(const unsigned int level) const {
    return IntVect{geom[level].Domain().smallEnd()};
}

/**
  * getDomainHi gets the upper bound of a given level in index space.
  *
  * @return An int vector: <xhi, yhi, zhi>
  */
IntVect    GridAmrex::getDomainHi(const unsigned int level) const {
    return IntVect{geom[level].Domain().bigEnd()};
}


/**
  * getProbLo gets the physical lower boundary of the domain.
  *
  * @return A real vector: <xlo, ylo, zlo>
  */
RealVect    GridAmrex::getProbLo() const {
    return RealVect{geom[0].ProbLo()};
}

/**
  * getProbHi gets the physical upper boundary of the domain.
  *
  * @return A real vector: <xhi, yhi, zhi>
  */
RealVect    GridAmrex::getProbHi() const {
    return RealVect{geom[0].ProbHi()};
}

/**
  * getMaxRefinement returns the maximum possible refinement level which was
  * specified by the user.
  *
  * @return Maximum (finest) refinement level of simulation.
  */
unsigned int GridAmrex::getMaxRefinement() const {
    return max_level;
}

/**
  * getMaxLevel returns the highest level of blocks actually in existence.
  *
  * @return The max level of existing blocks (0 is coarsest).
  */
unsigned int GridAmrex::getMaxLevel() const {
    return finest_level;
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
#ifdef GRID_LOG
    std::string msg = "[GridAmrex] Writing to plotfile: "+filename+"...";
    Logger::instance().log(msg);
#endif

    amrex::Vector<std::string>  names_amrex{names.size()};
    for (auto j=0; j<names.size(); ++j) {
        names_amrex[j] = names[j];
    }

    amrex::Vector<const amrex::MultiFab*> mfs;
    for(int i=0; i<=finest_level; ++i) {
        mfs.push_back( &unk_[i] );
    }
    amrex::Vector<int> lsteps(max_level+1 , 0);

    amrex::WriteMultiLevelPlotfile(filename,
                                   finest_level+1,
                                   mfs,
                                   names_amrex,
                                   geom,
                                   0.0,
                                   lsteps,
                                   ref_ratio);
}

/**
  *
  */
std::unique_ptr<TileIter> GridAmrex::buildTileIter(const unsigned int level) {
    return std::unique_ptr<TileIter>{new TileIterAmrex(unk_[level], level)};
}


/**
  * getDeltas gets the cell size for a given level.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  */
RealVect    GridAmrex::getDeltas(const unsigned int level) const {
    return RealVect{geom[level].CellSize()};
}


/** getCellFaceAreaLo gets lo face area of a cell with given integer coordinates
  *
  * @param axis Axis of desired face, returns the area of the lo side.
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return area of face (Real)
  */
Real  GridAmrex::getCellFaceAreaLo(const unsigned int axis,
                                   const unsigned int level,
                                   const IntVect& coord) const {
    return geom[level].AreaLo(amrex::IntVect(coord), axis);
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param level Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  */
Real  GridAmrex::getCellVolume(const unsigned int level,
                               const IntVect& coord) const {
    return geom[level].Volume(amrex::IntVect(coord));
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
  * @param level Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @returns The coordinates as a Fortran-style array.
  *
  * \todo profile this, see if we can get a version that doesn't require
  * extra copying.
  */
FArray1D    GridAmrex::getCellCoords(const unsigned int axis,
                                     const unsigned int edge,
                                     const unsigned int level,
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
    if (axis >= MILHOJA_NDIM) {
        // TODO: What value to put here?  Should it change
        //       based on edge?
        coords(idxLo) = 0.0;
        return coords;
    }

    //coordvec is length nElements + 1 if edge is Left or Right
    amrex::Vector<amrex::Real> coordvec;
    if        (edge == Edge::Left) {
        geom[level].GetEdgeLoc(coordvec, range, axis);
    } else if (edge == Edge::Right) {
        offset = 1;
        geom[level].GetEdgeLoc(coordvec, range, axis);
    } else if (edge == Edge::Center) {
        geom[level].GetCellLoc(coordvec, range, axis);
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
  * @param level Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param areaPtr Real Ptr to some fortran-style data structure. Will be filled
  *                with areas. Should be of shape:
  *                    (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  */
void    GridAmrex::fillCellFaceAreasLo(const unsigned int axis,
                                       const unsigned int level,
                                       const IntVect& lo,
                                       const IntVect& hi,
                                       Real* areaPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("GridAmrex::fillCellFaceAreasLo: Invalid axis.");
    }
#endif
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox area_fab{range, 1, areaPtr};
    geom[level].CoordSys::SetFaceArea(area_fab, range, axis);
}


/** fillCellVolumes fills a Real array (passed by pointer) with the
  * volumes of cells in a given range
  *
  * @param level Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param volPtr Real Ptr to some fortran-style data structure. Will be filled
  *             with volumes. Should be of shape:
  *                 (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  */
void    GridAmrex::fillCellVolumes(const unsigned int level,
                                   const IntVect& lo,
                                   const IntVect& hi,
                                   Real* volPtr) const {
    amrex::Box range{ amrex::IntVect(lo), amrex::IntVect(hi) };
    amrex::FArrayBox vol_fab{range, 1, volPtr};
    geom[level].CoordSys::SetVolume(vol_fab, range);
}

/**
 *
 */
void GridAmrex::fillPatch(amrex::MultiFab& mf, const int level) {
    if (level == 0) {
        amrex::Vector<amrex::MultiFab*>     smf;
        amrex::Vector<amrex::Real>          stime;
        smf.push_back(&unk_[0]);
        stime.push_back(0.0_wp);

        amrex::CpuBndryFuncFab      bndry_func(nullptr);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            physbc(geom[level], bcs_, bndry_func);

        amrex::FillPatchSingleLevel(mf, 0.0_wp, smf, stime,
                                    0, 0, mf.nComp(),
                                    geom[level], physbc, 0);
    }
    else {
        amrex::Vector<amrex::MultiFab*>     cmf;
        amrex::Vector<amrex::MultiFab*>     fmf;
        amrex::Vector<amrex::Real>          ctime;
        amrex::Vector<amrex::Real>          ftime;
        cmf.push_back(&unk_[level-1]);
        ctime.push_back(0.0_wp);
        fmf.push_back(&unk_[level]);
        ftime.push_back(0.0_wp);

        amrex::CpuBndryFuncFab      bndry_func(nullptr);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            cphysbc(geom[level-1], bcs_, bndry_func);
        amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
            fphysbc(geom[level  ], bcs_, bndry_func);

        // CellConservativeLinear interpolator from AMReX_Interpolator.H
        amrex::Interpolater* mapper = &amrex::cell_cons_interp;

        amrex::FillPatchTwoLevels(mf, 0.0_wp,
                                  cmf, ctime, fmf, ftime,
                                  0, 0, mf.nComp(),
                                  geom[level-1], geom[level],
                                  cphysbc, 0, fphysbc, 0,
                                  ref_ratio[level-1], mapper, bcs_, 0);
    }
}

//----- amrex::AmrCore OVERRIDES
/**
  * \brief Clear level
  *
  * \param level   Level being cleared
  */
void    GridAmrex::ClearLevel(int level) {
#ifdef GRID_LOG
    std::string msg = "[GridAmrex::ClearLevel] Clearing level " +
                      std::to_string(level) + "...";
    Logger::instance().log(msg);
#endif

    unk_[level].clear();
}

/**
  * \brief Remake Level
  *
  * \param level Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void   GridAmrex::RemakeLevel(int level, amrex::Real time,
                              const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[GridAmrex::RemakeLevel] Remaking level " +
                      std::to_string(level) + "...";
    Logger::instance().log(msg);
#endif

    amrex::MultiFab unkTmp{ba, dm, nCcVars_, nGuard_};
    fillPatch(unkTmp, level);

    std::swap(unkTmp, unk_[level]);
}

/**
  * \brief Make new level from scratch
  *
  * \todo Simulations should be allowed to use the GPU for setting the ICs.
  * Therefore, we need a means for expressing if the CPU-only or GPU-only thread
  * team configuration should be used.  If the GPU-only configuration is
  * allowed, then we should allow for more than one distributor thread.
  * \todo Use the runtime directly rather than recreate a thread team
  * configuration.
  * \todo The RuntimeAction should be created/set when initDomain is called
  *       and destroyed so that we get a failure here if this is used after
  *       initDomain finishes.  Similarly, this should fail in an obvious way if
  *       initDomain has not been called.
  * \todo Should this do a GC fill at the end?
  *
  * \param level Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void    GridAmrex::MakeNewLevelFromScratch(int level, amrex::Real time,
                                           const amrex::BoxArray& ba,
                                           const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[GridAmrex::MakeNewLevelFromScratch] Creating level " +
                      std::to_string(level) + "...";
    Logger::instance().log(msg);
#endif

    unk_[level].define(ba, dm, nCcVars_, nGuard_);
    unk_[level].setVal(0.0_wp);

    if (nThreads_initBlock_ <= 0) {
        throw std::invalid_argument("[GridAmrex::MakeNewLevelFromScratch] "
                                    "N computation threads must be positive");
    } else if (nDistributorThreads_initBlock_ != 1) {
        throw std::invalid_argument("[GridAmrex::MakeNewLevelFromScratch] "
                                    "Only one distributor thread presently allowed");
    } else if (nDistributorThreads_initBlock_ > nThreads_initBlock_) {
        throw std::invalid_argument("[GridAmrex::MakeNewLevelFromScratch] "
                                    "More distributor threads than computation threads");
    }

    RuntimeAction    action;
    action.name = "initBlock";
    action.nInitialThreads = nThreads_initBlock_ - nDistributorThreads_initBlock_;
    action.teamType = ThreadTeamDataType::BLOCK;
    action.routine = initBlock_;
    ThreadTeam  team(nThreads_initBlock_, 1);
    team.startCycle(action, "Cpu");

    // Initalize simulation block data in unk_[lev].
    Grid& grid = Grid::instance();
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        team.enqueue( ti->buildCurrentTile() );
    }
    team.closeQueue(nullptr);
    team.increaseThreadCount(nDistributorThreads_initBlock_);
    team.wait();

    // DO A GC FILL HERE?

#ifdef GRID_LOG
    msg =    "[GridAmrex::MakeNewLevelFromScratch] Created level "
           + std::to_string(level) + " with "
           + std::to_string(ba.size()) + " blocks.";
    Logger::instance().log(msg);
#endif
}

/**
  * \brief Make New Level from Coarse
  *
  * \todo Fail if initDomain has not yet been called.
  *
  * \param level Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void   GridAmrex::MakeNewLevelFromCoarse(int level, amrex::Real time,
                                         const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm) {
#ifdef GRID_LOG
    std::string msg = "[GridAmrex::MakeNewLevelFromCoarse] Making level " +
                      std::to_string(level) + " from coarse...";
    Logger::instance().log(msg);
#endif

    unk_[level].define(ba, dm, nCcVars_, nGuard_);

    amrex::CpuBndryFuncFab  bndry_func(nullptr);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
        cphysbc(geom[level-1], bcs_, bndry_func);
    amrex::PhysBCFunct<amrex::CpuBndryFuncFab>
        fphysbc(geom[level  ], bcs_, bndry_func);

    // CellConservativeLinear interpolator from AMReX_Interpolator.H
    amrex::Interpolater* mapper = &amrex::cell_cons_interp;

    amrex::InterpFromCoarseLevel(unk_[level], 0.0_wp, unk_[level-1],
                                 0, 0, nCcVars_,
                                 geom[level-1], geom[level],
                                 cphysbc, 0, fphysbc, 0,
                                 ref_ratio[level-1], mapper, bcs_, 0);
}

/**
  * \brief Tag boxes for refinement
  *
  * \todo Use tiling here?
  *
  * \param lev Level being checked
  * \param tags Tags for Box array
  * \param time Simulation time
  * \param ngrow ngrow
  */
void    GridAmrex::ErrorEst(int level, amrex::TagBoxArray& tags,
                            amrex::Real time, int ngrow) {
#ifdef GRID_LOG
    std::string msg = "[GridAmrex::ErrorEst] Doing ErrorEst for level " +
                      std::to_string(level) + "...";
    Logger::instance().log(msg);
#endif

    if (!errorEst_) {
        throw std::invalid_argument("[GridAmrex::ErrorEst] "
                                    "Error estimation pointer null");
    }

    amrex::Vector<int> itags;

    Grid& grid = Grid::instance();
    for (auto ti=grid.buildTileIter(level); ti->isValid(); ti->next()) {
        std::shared_ptr<Tile>   tileDesc = ti->buildCurrentTile();

        amrex::Box validbox{ amrex::IntVect(tileDesc->lo()),
                             amrex::IntVect(tileDesc->hi()) };
        amrex::TagBox&      tagfab = tags[tileDesc->gridIndex()];
        tagfab.get_itags(itags, validbox);

        //errorEst_(lev, tags, time, ngrow, tileDesc);
        int* tptr = itags.dataPtr();
        errorEst_(tileDesc, tptr);

        tagfab.tags_and_untags(itags,validbox);
    }

#ifdef GRID_LOG
    msg =   "[GridAmrex::ErrorEst] Did ErrorEst for level "
          + std::to_string(level) + ".";
    Logger::instance().log(msg);
#endif
}

}

