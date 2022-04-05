#include "Milhoja_GridAmrex.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <AMReX.H>
#include <AMReX_CoordSys.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_FillPatchUtil.H>

#include "Milhoja_Logger.h"
#include "Milhoja_GridConfiguration.h"
#include "Milhoja_axis.h"
#include "Milhoja_edge.h"
#include "Milhoja_TileIterAmrex.h"
#include "Milhoja_Runtime.h"

namespace milhoja {

//----- STATIC DATA MEMBER DEFAULT VALUES
bool GridAmrex::domainInitialized_ = false;
bool GridAmrex::domainDestroyed_   = false;

/**
  * Construct the AMReX Grid backend singleton.  It is assumed that
  * configuration values have already been loaded into the GridConfiguration
  * AMReX backend and that AMReX has been initialized.  The latter constraint is
  * required by the AmrCore base class' constructor.  When construction ends,
  * AMReX will be fully initialized.  However, the data structures needed to
  * store data will not have been created.
  *
  * The prior loading of the GridConfiguration singleton implies that
  * rudimentary validation of configuration values has been carried out.
  * Therefore, only minimal error checking need be done here.
  *
  * \todo Within local-scope block, retroactively check cast to int of
  * cfg.nGuard and cfg.nCcVars for overflow and fail if the stored values are
  * invalid.  We are forced to cast and then check since we want
  * nGuard_/nCcVars_ to be const.
  *  \todo Flux work is verbose and paranoid for development.  Simplify once we
  *        have more confidence in the implementation.
  */
GridAmrex::GridAmrex(void)
    : Grid(),
      AmrCore(),
      comm_{GridConfiguration::instance().mpiComm},
      nBlocksX_{GridConfiguration::instance().nBlocksX},
      nBlocksY_{GridConfiguration::instance().nBlocksY},
      nBlocksZ_{GridConfiguration::instance().nBlocksZ},
      nxb_{GridConfiguration::instance().nxb}, 
      nyb_{GridConfiguration::instance().nyb}, 
      nzb_{GridConfiguration::instance().nzb},
      nGuard_{static_cast<int>(GridConfiguration::instance().nGuard)},
      nCcVars_{static_cast<int>(GridConfiguration::instance().nCcVars)},
      nFluxVars_{static_cast<int>(GridConfiguration::instance().nFluxVars)},
      errorEst_{GridConfiguration::instance().errorEstimation},
      initBlock_noRuntime_{nullptr},
      initCpuAction_{}
{
    std::string   msg = "[GridAmrex] Initializing...";
    Logger&   logger = Logger::instance();
    logger.log(msg);

    // Satisfy grid configuration requirements and suggestions (See dev guide).
    {
        GridConfiguration&  cfg = GridConfiguration::instance();

        bcs_.resize(1);
        for (unsigned int i=0; i<MILHOJA_NDIM; ++i) {
            if        (cfg.loBCs[i] == BCs::Periodic) {
                bcs_[0].setLo(i, amrex::BCType::int_dir);
            } else if (cfg.loBCs[i] == BCs::External) {
//                bcs_[0].setLo(i, amrex::BCType::ext_dir);
                throw std::invalid_argument("[GridAmrex::GridAmrex] External BCs not supported yet");
            }
            if        (cfg.hiBCs[i] == BCs::Periodic) {
                bcs_[0].setHi(i, amrex::BCType::int_dir);
            } else if (cfg.hiBCs[i] == BCs::External) {
//                bcs_[0].setHi(i, amrex::BCType::ext_dir);
                throw std::invalid_argument("[GridAmrex::GridAmrex] External BCs not supported yet");
            }
        }

        cfg.clear();
    }

    // Set nonsensical values so that code will fail in obvious way if this is
    // accidentally used before initDomain is called.
    initCpuAction_.name = "initCpuAction_ used in ERROR";
    initCpuAction_.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    initCpuAction_.nInitialThreads =    0;
    initCpuAction_.nTilesPerPacket = 1000;
    initCpuAction_.routine         = nullptr;

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

    // Allocate and resize MultiFabs
    // TileIterAmrex stores references to unk_ and fluxes_ at a given level.
    // Therefore, these MFab arrays and the fluxes_[level] array must be set
    // here and remain fixed until finalization of the Grid unit.
    unk_.resize(max_level + 1);
    if (unk_.size() == 0) {
        throw std::logic_error("[GridAmrex::GridAmrex] CC data array emtpy");
    }
    msg = "[GridAmrex] Created " + std::to_string(unk_.size()) + " empty CC MultiFabs";
    logger.log(msg);

    // When constructing a TileIterAmrex at a given level, we must pass in a
    // vector of flux MultiFabs for the level.  If there are no flux variables,
    // an empty vector is needed.  Therefore, always allocate this outer vector.
    fluxes_.resize(max_level + 1);
    for (auto level=0; level<fluxes_.size(); ++level) {
        if (fluxes_[level].size() != 0) {
            throw std::logic_error("[GridAmrex::GridAmrex] Flux arrays aren't emtpy");
        }
    }
    if (nFluxVars_ > 0) {
        for (auto level=0; level<fluxes_.size(); ++level) {
            fluxes_[level].resize(MILHOJA_NDIM);
            msg =   "[GridAmrex] Created " + std::to_string(fluxes_[level].size())
                  + " empty flux MultiFabs at level " + std::to_string(level);
            logger.log(msg);
        }
    } else {
        logger.log("[GridAmrex] No flux MultiFabs needed");
    }

    //----- LOG GRID CONFIGURATION INFORMATION
    int size = -1;
    MPI_Comm_size(comm_, &size);

    // Get values owned by AMReX
    RealVect  domainLo = getProbLo();
    RealVect  domainHi = getProbHi();

    msg =   "[GridAmrex] " + std::to_string(size)
          + " MPI processes in given communicator";
    logger.log(msg);

    msg = "[GridAmrex] N dimensions = " + std::to_string(MILHOJA_NDIM);
    logger.log(msg);

    msg = "[GridAmrex] ";
    switch (getCoordinateSystem()) {
    case CoordSys::Cartesian:
        msg += "Cartesian";
        break;
    case CoordSys::Cylindrical:
        msg += "Cylindrical";
        break;
    case CoordSys::Spherical:
        msg += "Spherical";
        break;
    default:
        throw std::logic_error("[GridAmrex::GridAmrex] Unknown coordinate system");
    }
    msg += " Coordinate System";
    logger.log(msg);

    msg  = "[GridAmrex] Physical spatial domain specification";
    logger.log(msg);
    msg =   "[GridAmrex]    x in ("
          + std::to_string(domainLo[Axis::I]) + ", "
          + std::to_string(domainHi[Axis::I]) + ")";
    logger.log(msg);
#if MILHOJA_NDIM >= 2
    msg =   "[GridAmrex]    y in ("
          + std::to_string(domainLo[Axis::J]) + ", "
          + std::to_string(domainHi[Axis::J]) + ")";
    logger.log(msg);
#endif
#if MILHOJA_NDIM >= 3
    msg =   "[GridAmrex]    z in ("
          + std::to_string(domainLo[Axis::K]) + ", "
          + std::to_string(domainHi[Axis::K]) + ")";
    logger.log(msg);
#endif

    msg = "[GridAmrex] Boundary Conditions Handling";
    logger.log(msg);
    for (unsigned int i=0; i<MILHOJA_NDIM; ++i) {
        if        (i == Axis::I) {
            msg = "[GricAmrex] X-axis ";
        } else if (i == Axis::J) {
            msg = "[GricAmrex] Y-axis ";
        } else if (i == Axis::K) {
            msg = "[GricAmrex] Z-axis ";
        }

        if        (bcs_[0].lo(i) == amrex::BCType::int_dir) {
            logger.log(msg + "lo = Periodic");
        } else if (bcs_[0].lo(i) == amrex::BCType::ext_dir) {
            logger.log(msg + "lo = External");
        } else {
            throw std::logic_error("[GridAmrex::GridAmrex] Invalid lo AMReX BC type");
        }

        if        (bcs_[0].hi(i) == amrex::BCType::int_dir) {
            logger.log(msg + "hi = Periodic");
        } else if (bcs_[0].hi(i) == amrex::BCType::ext_dir) {
            logger.log(msg + "hi = External");
        } else {
            throw std::logic_error("[GridAmrex::GridAmrex] Invalid hi AMReX BC type");
        }
    }

    msg =   "[GridAmrex] Maximum Finest Level = "
          + std::to_string(max_level);
    logger.log(msg);

    msg =   "[GridAmrex] N Guardcells = "
          + std::to_string(nGuard_);
    logger.log(msg);

    msg =   "[GridAmrex] N Cell-centered Variables = "
          + std::to_string(nCcVars_);
    logger.log(msg);

    msg =   "[GridAmrex] N Flux Variables = "
          + std::to_string(nFluxVars_);
    logger.log(msg);

#if   MILHOJA_NDIM == 1
    msg =   "[GridAmrex] Block interior size = "
          + std::to_string(nxb_)
          + " cells";
    logger.log(msg);

    msg =   "[GridAmrex] Domain decomposition at coarsest level = "
          + std::to_string(nBlocksX_)
          + " blocks";
    logger.log(msg);
    
    msg = "[GridAmrex] Mesh deltas by level";
    logger.log(msg);
    for (int level=0; level<=max_level; ++level) {
        RealVect  deltas = getDeltas(level);
        msg =   "[GridAmrex]    Level " + std::to_string(level)
              + "       "
              + std::to_string(deltas[Axis::I]);
        logger.log(msg);
    }
#elif MILHOJA_NDIM == 2
    msg =   "[GridAmrex] Block interior size = "
          + std::to_string(nxb_) + " x "
          + std::to_string(nyb_)
          + " cells";
    logger.log(msg);

    msg =   "[GridAmrex] Domain decomposition at coarsest level = "
          + std::to_string(nBlocksX_) + " x "
          + std::to_string(nBlocksY_)
          + " blocks";
    logger.log(msg);

    msg = "[GridAmrex] Mesh deltas by level";
    logger.log(msg);
    for (int level=0; level<=max_level; ++level) {
        RealVect  deltas = getDeltas(level);
        msg =   "[GridAmrex]    Level " + std::to_string(level)
              + "       "
              + std::to_string(deltas[Axis::I]) + " x "
              + std::to_string(deltas[Axis::J]);
        logger.log(msg);
    }
#elif MILHOJA_NDIM == 3
    msg =   "[GridAmrex] Block interior size = "
          + std::to_string(nxb_) + " x "
          + std::to_string(nyb_) + " x "
          + std::to_string(nzb_)
          + " cells";
    logger.log(msg);

    msg =   "[GridAmrex] Domain decomposition at coarsest level = "
          + std::to_string(nBlocksX_) + " x "
          + std::to_string(nBlocksY_) + " x "
          + std::to_string(nBlocksZ_)
          + " blocks";
    logger.log(msg);

    msg = "[GridAmrex] Mesh deltas by level";
    logger.log(msg);
    for (int level=0; level<=max_level; ++level) {
        RealVect  deltas = getDeltas(level);
        msg =   "[GridAmrex]    Level " + std::to_string(level)
              + "       "
              + std::to_string(deltas[Axis::I]) + " x "
              + std::to_string(deltas[Axis::J]) + " x "
              + std::to_string(deltas[Axis::K]);
        logger.log(msg);
    }
#endif

    logger.log("[GridAmrex] Created and ready for use");
}

/**
  * Under normal program execution and if initialize been called, it is a
  * logical error for singleton destruction to occur without the calling code
  * having first called finalize.
  */
GridAmrex::~GridAmrex(void) {
    if ((initialized_) && (!finalized_)) {
        std::cerr << "[GridAmrex::~GridAmrex] ERROR - Grid not finalized"
                  << std::endl;
    }
}

/**
 *  Finalize the Grid singleton by cleaning up all AMReX resources and
 *  finalizing AMReX.  This must be called before MPI is finalized.
 *
 *  \todo Flux work is verbose and paranoid for development.  Simplify once we
 *        have more confidence in the implementation.
 */
void  GridAmrex::finalize(void) {
    // We need to do error checking explicitly upfront rather than wait for the
    // Grid base implementation to do so; else, we would finalize AMReX twice.
    if ((domainInitialized_) && (!domainDestroyed_)) {
        throw std::logic_error("[GridAmrex::finalize] Domain not destroyed");
    } else if (!initialized_) {
        throw std::logic_error("[GridAmrex::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[GridAmrex::finalize] Already finalized");
    }

    Logger::instance().log("[GridAmrex] Finalizing ...");

    // Clean-up all AMReX structures before finalization.
    std::string    msg{};
    msg =   "[GridAmrex] Destroying cell-centered MultiFab array with "
          + std::to_string(unk_.size()) + " level(s)";
    Logger::instance().log(msg);
    std::vector<amrex::MultiFab>().swap(unk_);
    if (unk_.size() != 0) {
        throw std::runtime_error("[GridAmrex::finalize] Didn't destroy CC array");
    }

    for (auto level=0; level<fluxes_.size(); ++level) {
        if        ((nFluxVars_ == 0) && (fluxes_[level].size() >  0)) {
            throw std::logic_error("[GridAmrex::finalize] Flux multifabs created");
        } else if ((nFluxVars_ >  0) && (fluxes_[level].size() == 0)) {
            throw std::logic_error("[GridAmrex::finalize] Flux multifabs not created");
        } else if ((nFluxVars_ >  0) && (fluxes_[level].size() >  0)) {
            msg =   "[GridAmrex] Destroying " + std::to_string(fluxes_[level].size())
                  + " flux MultiFabs at level " + std::to_string(level);
            Logger::instance().log(msg);
            std::vector<amrex::MultiFab>().swap(fluxes_[level]);
            if (fluxes_[level].size() != 0) {
                throw std::runtime_error("[GridAmrex::finalize] Didn't destroy flux MFab array");
            }
        }
    }
    msg =   "[GridAmrex] Destroying flux array with "
          + std::to_string(fluxes_.size()) + " level(s)";
    Logger::instance().log(msg);
    std::vector<std::vector<amrex::MultiFab>>().swap(fluxes_);
    if (fluxes_.size() != 0) {
        throw std::runtime_error("Didn't destroy flux array");
    }

    // This is the ugliest part of the multiple inheritance design of this class
    // because we finalize AMReX before AmrCore is destroyed, which occurs
    // whenever the compiler decides to destroy the Grid backend singleton.
    amrex::Finalize();

    Grid::finalize();

    Logger::instance().log("[GridAmrex] Finalized");
}

/**
 *  Destroy the domain.  It is a logical error to call this if initDomain has
 *  not already been called or to call it multiple times.
 */
void  GridAmrex::destroyDomain(void) {
    if (!domainInitialized_) {
        throw std::logic_error("[GridAmrex::destroyDomain] initDomain never called");
    } else if (domainDestroyed_) {
        throw std::logic_error("[GridAmrex::destroyDomain] destroyDomain already called");
    }

    // Set nonsensical value for error estimation routine so that accidental use
    // of the pointer will fail in obvious way.
    errorEst_ = nullptr;
    domainDestroyed_ = true;

    Logger::instance().log("[GridAmrex] Destroyed domain.");
}

/**
 * Set the initial conditions and setup the grid structure so that the initial
 * conditions are resolved in accord with the Grid configuration.  This is
 * executed one block at a time without the runtime.
 *
 */
void    GridAmrex::initDomain(ACTION_ROUTINE initBlock) {
    // domainDestroyed_ => domainInitialized_
    // Therefore, no need to check domainDestroyed_.
    if (domainInitialized_) {
        throw std::logic_error("[GridAmrex::initDomain] initDomain already called");
    } else if (!initBlock) {
        throw std::invalid_argument("[GridAmrex::initDomain] null initBlock pointer");
    }

    Logger::instance().log("[GridAmrex] Initializing domain...");

    initBlock_noRuntime_ = initBlock;
    InitFromScratch(0.0_wp);
    initBlock_noRuntime_ = nullptr;

    domainInitialized_ = true;

    std::vector<amrex::MultiFab>::size_type   nGlobalBlocks = 0;
    for (int level=0; level<=finest_level; ++level) {
        nGlobalBlocks += unk_[level].size();
    }

    std::string msg = "[GridAmrex] Initialized domain with " +
                      std::to_string(nGlobalBlocks) +
                      " total blocks.";
    Logger::instance().log(msg);
}

/**
 * Set the initial conditions and setup the grid structure so that the initial
 * conditions are resolved in accord with the Grid configuration.  This is
 * carried out by the runtime using the CPU-only thread team configuration.
 */ 
void GridAmrex::initDomain(const RuntimeAction& cpuAction) {
    // domainDestroyed_ => domainInitialized_
    // Therefore, no need to check domainDestroyed_.
    if (domainInitialized_) {
        throw std::logic_error("[GridAmrex::initDomain] initDomain already called");
    }

    Logger::instance().log("[GridAmrex] Initializing domain with runtime...");

    // Cache given action so that AmrCore routines can access action.
    initCpuAction_ = cpuAction;

    InitFromScratch(0.0_wp);

    // Set nonsensical values so that code will fail in obvious way if this is
    // accidentally used in the future.
    initCpuAction_.name = "initCpuAction_ used in ERROR";
    initCpuAction_.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
    initCpuAction_.nInitialThreads =    0;
    initCpuAction_.nTilesPerPacket = 1000;
    initCpuAction_.routine         = nullptr;

    domainInitialized_ = true;

    std::vector<amrex::MultiFab>::size_type   nGlobalBlocks = 0;
    for (int level=0; level<=finest_level; ++level) {
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
    for (int level=0; level<=finest_level; ++level) {
#ifdef GRID_LOG
        Logger::instance().log(  "[GridAmrex] GCFill on level "
                               + std::to_string(level) );
#endif

        fillPatch(unk_[level], level);
    }
}

/**
 * Obtain the size of the interior of all blocks.
 *
 * NOTE: This function does not presume to know what values to set for variables
 * above MILHOJA_NDIM.  Therefore, calling code is responsible for setting or
 * ignoring such data.  This routine will not alter or overwrite such variables.
 */
void    GridAmrex::getBlockSize(unsigned int* nxb,
                                unsigned int* nyb,
                                unsigned int* nzb) const {
    if (!nxb) {
        std::string    msg = "[GridAmrex::getBlockSize] nxb null";
        throw std::invalid_argument(msg);
    }
    *nxb = nxb_;
#if MILHOJA_NDIM >= 2
    if (!nyb) {
        std::string    msg = "[GridAmrex::getBlockSize] nyb null";
        throw std::invalid_argument(msg);
    }
    *nyb = nyb_;
#endif
#if MILHOJA_NDIM == 3
    if (!nzb) {
        std::string    msg = "[GridAmrex::getBlockSize] nzb null";
        throw std::invalid_argument(msg);
    }
    *nzb = nzb_;
#endif
}

/**
 * Obtain the block decomposition of the domain on the coarsest level.
 *
 * NOTE: This function does not presume to know what values to set for variables
 * above MILHOJA_NDIM.  Therefore, calling code is responsible for setting or
 * ignoring such data.  This routine will not alter or overwrite such variables.
 */
void    GridAmrex::getDomainDecomposition(unsigned int* nBlocksX,
                                          unsigned int* nBlocksY,
                                          unsigned int* nBlocksZ) const {
    if (!nBlocksX) {
        std::string    msg = "[GridAmrex::getDomainDecomposition] nBlocksX null";
        throw std::invalid_argument(msg);
    }
    *nBlocksX = nBlocksX_;
#if MILHOJA_NDIM >= 2
    if (!nBlocksY) {
        std::string    msg = "[GridAmrex::getDomainDecomposition] nBlocksY null";
        throw std::invalid_argument(msg);
    }
    *nBlocksY = nBlocksY_;
#endif
#if MILHOJA_NDIM == 3
    if (!nBlocksZ) {
        std::string    msg = "[GridAmrex::getDomainDecomposition] nBlocksZ null";
        throw std::invalid_argument(msg);
    }
    *nBlocksZ = nBlocksZ_;
#endif
}

/**
  * Obtain the coordinate system used to define the domain.
  */
CoordSys    GridAmrex::getCoordinateSystem(void) const {
    switch(geom[0].Coord()) {
    case amrex::CoordSys::CoordType::cartesian:
        return CoordSys::Cartesian;
    case amrex::CoordSys::CoordType::RZ:
        return CoordSys::Cylindrical;
    case amrex::CoordSys::CoordType::SPHERICAL:
        return CoordSys::Spherical;
    default:
        throw std::logic_error("[GridAmrex::getCoordinateSystem] Unknown system");
    }
}

/**
  * getDomainLo gets the lower bound of a given level index space.
  *
  * \todo Sanity check level value
  *
  * @return An int vector: <xlo, ylo, zlo>
  */
IntVect    GridAmrex::getDomainLo(const unsigned int level) const {
    return IntVect{geom[level].Domain().smallEnd()};
}

/**
  * getDomainHi gets the upper bound of a given level in index space.
  *
  * \todo Sanity check level value
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
  * \todo Rename to match the C/Fortran interface name.
  *
  * @return Maximum (finest) refinement level of simulation.
  */
unsigned int GridAmrex::getMaxRefinement() const {
    if (max_level < 0) {
        throw std::logic_error("[GridAmrex::getMaxRefinement] max_level negative");
    }
    return static_cast<unsigned int>(max_level);
}

/**
  * getMaxLevel returns the highest level of blocks actually in existence.
  *
  * \todo Rename to match the C/Fortran interface name.
  *
  * @return The max level of existing blocks (0 is coarsest).
  */
unsigned int GridAmrex::getMaxLevel() const {
    if (finest_level < 0) {
        throw std::logic_error("[GridAmrex::getMaxLevel] finest_level negative");
    }
    return static_cast<unsigned int>(finest_level);
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

    Logger::instance().log("[GridAmrex] Wrote to plotfile " + filename);
}

/**
  *
  * \todo Sanity check level value
  *
  */
std::unique_ptr<TileIter> GridAmrex::buildTileIter(const unsigned int level) {
    return std::unique_ptr<TileIter>{new TileIterAmrex{unk_[level],
                                                       fluxes_[level],
                                                       level}};
}

/**
  * It is intended that this routine only be used in the Fortran/C++
  * interoperability layer, where the use of a unique_ptr will not work.
  *
  * This routine dynamically allocates the memory of the iterator object.
  * Calling code is responsible for using delete to release the resources once
  * the iterator is no longer needed.
  *
  * \todo Sanity check level value
  */
TileIter* GridAmrex::buildTileIter_forFortran(const unsigned int level) {
    return (new TileIterAmrex{unk_[level], fluxes_[level], level});
}

/**
  * getDeltas gets the cell size for a given level.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  */
RealVect    GridAmrex::getDeltas(const unsigned int level) const {
    if (level > max_level) {
        std::string    msg =   "[GridAmrex::getDeltas] Invalid level value "
                             + std::to_string(level);
        throw std::invalid_argument(msg);
    }

    return RealVect{geom[level].CellSize()};
}

/** getCellFaceAreaLo gets lo face area of a cell with given integer coordinates
  *
  * \todo Sanity check axis & level values
  *
  * @param axis Axis of desired face, returns the area of the lo side.
  * @param level Level (0-based)
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
  * \todo Sanity check level value
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
  * \todo Sanity check axis, edge, and level values.
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
  * \todo Sanity check axis and level values.
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
  * \todo Sanity check level value.
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
  * How does data get set?  We can get interpolation, but also for BCs.  How is
  * interpolation done and what method is used?
  *
  * \todo Sanity check level value.
  * \todo RemakeLevel seems to assume that this will also set the interior data.
  * Is this true?  If so, is this what we want?
  * \todo Does fillPatch require data in primitive form?  If so, document how,
  * when, and where the p-to-c and reverse transformations are done.
  * \todo Interpolation by AMReX's conservative linear interpolation routine?
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
  * AmrCore calls this function to destroy all MultiFabs managed by this class
  * at the given level.
  * 
  * It is intended that this function only be called by AMReX.
  *
  * \todo Sanity check level value.
  *
  * \param level   Level being cleared
  */
void    GridAmrex::ClearLevel(int level) {
    unk_[level].clear();

    if (nFluxVars_ > 0) {
        for (unsigned int i=0; i<fluxes_[level].size(); ++i) {
            fluxes_[level][i].clear();
        }
    }

    Logger::instance().log("[GridAmrex] Cleared level " + std::to_string(level));
}

/**
  * \brief Remake Level
  *
  * AmrCore calls this routine to reestablish the data in each MultiFab defined at
  * the given level (e.g., cell-centered, fluxes) onto a new MultiFab specified
  * through the given box array and distribution map.  The remade cell-centered
  * data MultiFab will have data in all interior cells as well as guardcells
  * via the fillPatch() function.
  *
  * It is intended that this function only be called by AMReX.
  *
  * Note that the new MultiFab might contain new boxes recently added to the
  * level.  Therefore fillPatch() might execute AMReX's conservative linear
  * interpolation algorithm to set data in this level based on data at the
  * coarser level.
  *
  * Since fillPatch() requires that data be in primitive form, this function
  * also has the same requirement.
  *
  * \todo Sanity check level value.
  *
  * \param level Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void   GridAmrex::RemakeLevel(int level, amrex::Real time,
                              const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm) {
    amrex::MultiFab unkTmp{ba, dm, nCcVars_, nGuard_};
    // Move all unk data (interior and GC) to given ba/dm layout.
    // Do *not* use sub-cycling.
    fillPatch(unkTmp, level);
    std::swap(unkTmp, unk_[level]);

    if (nFluxVars_ > 0) {
        assert(fluxes_[level].size() == MILHOJA_NDIM);
        for (unsigned int i=0; i<MILHOJA_NDIM; ++i) {
            fluxes_[level][i].define(amrex::convert(ba, amrex::IntVect::TheDimensionVector(i)),
                                     dm, nFluxVars_, NO_GC_FOR_FLUX);
        }
    }

    Logger::instance().log("[GridAmrex] Remade level " + std::to_string(level));
}

/**
  * \brief Make new level from scratch
  *
  * AmrCore calls this function so that for the given refinement level we can
  *  (1) create a MultiFab for each data type using the given box array and
  *      distribution map,
  *  (2) initialize all interior block data in the cell-centered MultiFab
  *      with initial conditions via the initBlock routine passed to
  *      initDomain(), and
  *  (3) fill all GCs in the cell-centered MultiFab.
  * The data in all flux MultiFabs is not initialized.
  *
  * Step (3) is required as refinement/derefinement routines can choose to
  * refine a block if they determine that there is poorly-refined data in the
  * GC.  In other words, this routine leaves the ICs ready for immediate use by
  * ErrorEst().
  *
  * While it is up to calling code to determine how and where to set the ICs,
  * the initBlock function must, at the very least, set all data that will be
  * used by ErrorEst to set the initial AMR refinement.
  *
  * \todo Clearly the initBlock routine should be capable of computing the ICs
  * in the GCs.  Why not just make it a requirement that these routines be
  * written in this way?  Note that Flash-X would have to change its
  * requirements so that Simulation_initBlock routines comply.  If we go this
  * way, then no IC data in the level would be set by interpolation.  That means
  * that we really can set the ICs and compute the EoS here in the same go.
  * \todo Simulations should be allowed to use the GPU for setting the ICs.
  * Therefore, we need a means for expressing if the CPU-only or GPU-only thread
  * team configuration should be used.  If the GPU-only configuration is
  * allowed, then we should allow for more than one distributor thread.
  * \todo Sanity check level value.
  *
  * \param level Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void    GridAmrex::MakeNewLevelFromScratch(int level, amrex::Real time,
                                           const amrex::BoxArray& ba,
                                           const amrex::DistributionMapping& dm) {
    unk_[level].define(ba, dm, nCcVars_, nGuard_);
    std::string   msg =   "[GridAmrex] Made CC MultiFab from scratch at level "
                        + std::to_string(level);
    Logger::instance().log(msg);

    if (nFluxVars_ > 0) {
        assert(fluxes_[level].size() == MILHOJA_NDIM);
        for (unsigned int i=0; i<MILHOJA_NDIM; ++i) {
            fluxes_[level][i].define(amrex::convert(ba, amrex::IntVect::TheDimensionVector(i)),
                                     dm, nFluxVars_, NO_GC_FOR_FLUX);
            msg =   "[GridAmrex] Made flux MultiFab " + std::to_string(i) 
                  + " from scratch at level " + std::to_string(level);
            Logger::instance().log(msg);
        }
    }

    if        ((!initBlock_noRuntime_) && ( initCpuAction_.routine)) {
        // Apply initial conditions using the runtime
        Runtime::instance().executeCpuTasks("MakeNewLevelFromScratch", initCpuAction_);
    } else if (( initBlock_noRuntime_) && (!initCpuAction_.routine)) {
        // Apply initial conditions using just the iterator
        for (auto ti = Grid::instance().buildTileIter(level); ti->isValid(); ti->next()) {
            std::unique_ptr<Tile> tileDesc = ti->buildCurrentTile();
            initBlock_noRuntime_(0, tileDesc.get());
        }
    } else if ((!initBlock_noRuntime_) && (!initCpuAction_.routine)) {
        throw std::logic_error("[GridAmres::MakeNewLevelFromScratch] No IC routine given");
    } else {
        throw std::logic_error("[GridAmres::MakeNewLevelFromScratch] Two IC routines given");
    }

    fillPatch(unk_[level], level);

    msg =   "[GridAmrex] Created level "
          + std::to_string(level) + " from scratch with "
          + std::to_string(ba.size()) + " blocks";
    Logger::instance().log(msg);
}

/**
  * \brief Make New Level from Coarse
  *
  * AmrCore calls this function so that for the given refinement level we can
  *  (1) create a MultiFab for each data type using the given box array and
  *      distribution map and
  *  (2) set all data in the interior and guardcells of the cell-centered data
  *      MultiFab using the data of the next coarsest level via fillPatch().
  * The data in the flux MultiFabs is not initialized.
  *
  * Since fillPatch() requires that data be in primitive form, this function
  * also has the same requirement.
  *
  * This routine should only be invoked by AMReX.
  *
  * \todo Fail if initDomain has not yet been called.
  * \todo Sanity check level value.
  *
  * \param level Level being made
  * \param time Simulation time
  * \param ba BoxArray of level being made
  * \param dm DistributionMapping of leving being made
  */
void   GridAmrex::MakeNewLevelFromCoarse(int level, amrex::Real time,
                                         const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm) {
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
    std::string   msg =   "[GridAmrex] Made CC MultiFab at level "
                        + std::to_string(level)
                        + " from data in coarse level";
    Logger::instance().log(msg);

    if (nFluxVars_ > 0) {
        assert(fluxes_[level].size() == MILHOJA_NDIM);
        for (unsigned int i=0; i<MILHOJA_NDIM; ++i) {
            fluxes_[level][i].define(amrex::convert(ba, amrex::IntVect::TheDimensionVector(i)),
                                     dm, nFluxVars_, NO_GC_FOR_FLUX);
            msg =   "[GridAmrex] Made Flux MultiFab " + std::to_string(i)
                  + " at level " + std::to_string(level)
                  + " from data in coarse level";
            Logger::instance().log(msg);
        }
    }

    msg =   "[GridAmrex] Created level "
          + std::to_string(level) + " from coarse level with "
          + std::to_string(ba.size()) + " blocks";
    Logger::instance().log(msg);
}

/**
  * \brief Tag boxes for refinement
  *
  * This function effectively wraps the error estimation routine configured into
  * the Grid class so that it can be used directly by AMReX.  In particular,
  * AmrCore may use this subroutine many times during the process of grid
  * refinement so that calling code may communicate which blocks in the given
  * level require refinement.  The final refinement decisions are made by AMReX
  * based on the information gathered with this callback.
  *
  * This routine iterates across all blocks in the given level and uses the
  * error estimation routine to determine if the current block needs refinement.
  *
  * \todo Use tiling here?
  * \todo Sanity check level value.
  *
  * \param level Level being checked
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

