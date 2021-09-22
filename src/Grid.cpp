#include "Grid.h"

#include <stdexcept>

#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "OrchestrationLogger.h"

// TODO: move to a header?
#ifdef GRID_AMREX
#include "GridAmrex.h"
namespace orchestration {
    typedef GridAmrex GridVersion;
}
#else
#error "Need to specify Grid implementation with GRID_[NAME] macro"
#endif

namespace orchestration {

bool           Grid::instantiated_ = false;
bool           Grid::finalized_ = false;
MPI_Comm       Grid::globalComm_ = MPI_COMM_NULL;
Real           Grid::xMin_       =  0.0;
Real           Grid::xMax_       = -1.0;
Real           Grid::yMin_       =  0.0;
Real           Grid::yMax_       = -1.0;
Real           Grid::zMin_       =  0.0;
Real           Grid::zMax_       = -1.0;
unsigned int   Grid::nxb_        =  0;
unsigned int   Grid::nyb_        =  0;
unsigned int   Grid::nzb_        =  0;
unsigned int   Grid::nBlocksX_   =  0;
unsigned int   Grid::nBlocksY_   =  0;
unsigned int   Grid::nBlocksZ_   =  0;
unsigned int   Grid::lRefineMax_ =  0;
unsigned int   Grid::nGuard_     =  0;

/**
 * instace gets a reference to the singleton Grid object.  This can only be used
 * after calling instantiate, and should not be used after calling finalize.
 *
 * @return A reference to the singleton object, which has been downcast
 *         to Grid type.
 */
Grid&   Grid::instance(void) {
    if (!instantiated_) {
        throw std::logic_error("[Grid::instance] Singleton not instantiated yet");
    } else if (finalized_) {
        throw std::logic_error("[Grid::instance] Singleton already finalized");
    }

    static GridVersion gridSingleton;
    return gridSingleton;
}

/**
 * instantiate allows the user to easily distinguish the first call to
 * instance, which calls the Grid constructor, from all subsequent calls.
 * It must be called exactly once in the program, before all calls to instance.
 * It should not be called after finalize has been called.
 */
void   Grid::instantiate(const MPI_Comm comm,
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
                         const unsigned int nGuard) {
    Logger::instance().log("[Grid] Initializing...");

    if (instantiated_) {
        throw std::logic_error("[Grid::instantiate] Singleton already instantiated");
    } else if (finalized_) {
        throw std::logic_error("[Grid::instantiate] Singleton already finalized");
    }

    globalComm_ = comm;

    // Set the internal static variables now as these communicate
    // setup values to the Grid implementation when it is constructed.
    xMin_ = xMin;    xMax_ = xMax;
    yMin_ = yMin;    yMax_ = yMax;
    zMin_ = zMin;    zMax_ = zMax;

    nxb_      = nxb;        nyb_      = nyb;        nzb_      = nzb;
    nBlocksX_ = nBlocksX;   nBlocksY_ = nBlocksY;   nBlocksZ_ = nBlocksZ;

    lRefineMax_ = lRefineMax;
    nGuard_     = nGuard;

    instantiated_ = true;
    Grid::instance();

    // Since the setup values have been consumed and should never be used
    // again, set them to nonsense values.
    xMin_ = 0.0;    xMax_ = -1.0;
    yMin_ = 0.0;    yMax_ = -1.0;
    zMin_ = 0.0;    zMax_ = -1.0;

    nxb_      = 0;   nyb_      = 0;   nzb_      = 0;
    nBlocksX_ = 0;   nBlocksY_ = 0;   nBlocksZ_ = 0;

    lRefineMax_ = 0;
    nGuard_ = 0;

    Logger::instance().log("[Grid] Created and ready for use");
}

/**
 * We do not control when the Grid singleton is destroyed.  For instance, it
 * could be destroyed when the program ends and, therefore, after the program
 * calls MPI_Finalize.  If a Grid implementation needs to perform MPI-based
 * clean-up, then finalizing upon destruction would lead to incorrect execution.
 * Therefore, we allow for manually finalizing the Grid singleton.  This member
 * function must be called if instantiate has been called and should never be
 * called more than once.  Further, the Grid singleton should not be used once
 * this member function has been called.
 *
 * Derived classes that overload this implementation should call this member
 * function *after* performing its own clean-up.
 */
void   Grid::finalize(void) {
    Logger::instance().log("[Grid] Finalizing ...");

    if (!instantiated_) {
        throw std::logic_error("[Grid::finalize] Singleton never instantiated");
    } else if (finalized_) {
        throw std::logic_error("[Grid::finalize] Singleton already finalized");
    }

    globalComm_ = MPI_COMM_NULL;
    finalized_ = true;

    Logger::instance().log("[Grid] Finalized");
}

/**
  * getDeltas gets the cell size for a given level.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  * \todo default implementation
  */
RealVect    Grid::getDeltas(const unsigned int level) const {
    throw std::logic_error("Default Grid::getDeltas not yet implemented");
}

/** getCellFaceAreaLo gets lo face area of a cell with given integer coordinates
  *
  * @param axis Axis of desired face, returns the area of the lo side.
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return area of face (Real)
  * \todo default implementation
  */
Real  Grid::getCellFaceAreaLo(const unsigned int axis, const unsigned int lev, const IntVect& coord) const {
    throw std::logic_error("Default Grid::getCellFaceAreaLo not yet implemented");
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  * \todo default implementation
  */
Real  Grid::getCellVolume(const unsigned int lev, const IntVect& coord) const {
    throw std::logic_error("Default Grid::getCellVolume not yet implemented");
}

/** fillCellCoords fills a Real array (passed by pointer) with the
  * cell coordinates in a given range
  *
  * @param axis Axis of desired coord (allowed: Axis::{I,J,K})
  * @param edge Edge of desired coord (allowed: Edge::{Left,Right,Center})
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @return FArray1D wrapper around array of length hi[axis]-lo[axis]+1.
  * \todo default implementation
  */
FArray1D    Grid::getCellCoords(const unsigned int axis, const unsigned int edge,
                                const unsigned int lev, const IntVect& lo,
                                const IntVect& hi) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("Grid::getCellCoords: Invalid axis.");
    }
    if(edge!=Edge::Left && edge!=Edge::Right && edge!=Edge::Center){
        throw std::logic_error("Grid::getCellCoords: Invalid edge.");
    }
#endif
    throw std::logic_error("Default Grid::getCellCoords not yet implemented");
}

/** fillCellFaceAreasLo fills a Real array (passed by pointer) with the
  * cell face areas in a given range.
  *
  * @param axis Axis of desired coord (allowed: Axis::{I,J,K})
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param areaPtr Real Ptr to some fortran-style data structure. Will be filled with areas.
  *             Should be of shape (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  * \todo default implementation
  */
void    Grid::fillCellFaceAreasLo(const unsigned int axis,
                                  const unsigned int lev, const IntVect& lo,
                                  const IntVect& hi, Real* areaPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("Grid::fillCellFaceAreasLo: Invalid axis.");
    }
#endif
    throw std::logic_error("Default Grid::fillCellFaceAreasLo not yet implemented");
}


/** fillCellVolumes fills a Real array (passed by pointer) with the
  * volumes of cells in a given range
  *
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param volPtr Real Ptr to some fortran-style data structure. Will be filled with volumes.
  *             Should be of shape (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  * \todo default implementation
  */
void    Grid::fillCellVolumes(const unsigned int lev, const IntVect& lo,
                              const IntVect& hi, Real* volPtr) const {
    throw std::logic_error("Default Grid::fillCellVolumes not yet implemented");
}

/** subcellGeometry fills a Real array (passed by pointer) with the
  * volumes of subcells
  *
  * @param nsubI No. of subcell lengths per cell length in x-dir
  * @param nsubJ No. of subcell lengths per cell length in y-dir
  * @param nsubK No. of subcell lengths per cell length in z-dir
  * @param dvCell Volume of whole cell
  * @param dvSubPtr Volumes of subcells (should be length nsubI*nsubJ)
  * @param xL x-coord of left cell face (optional, default 0.0)
  * @param xR x-coord of right cell face (optional, default 0.0)
  * @param yL y-coord of lower cell face (optional, default 0.0)
  * @param yR y-coord of upper cell face (optional, default 0.0)
  *
  * \todo implement non-cartesian versions
  * \todo adjust interface with FArray?
  */
void    Grid::subcellGeometry(const unsigned int nsubI,
                              const unsigned int nsubJ,
                              const unsigned int nsubK,
                              const Real dvCell, Real* dvSubPtr,
                              const Real xL, const Real xR,
                              const Real yL, const Real yR) {
    Real denom = Real(nsubI * nsubJ * nsubK);
    Real dvs = dvCell / denom;
    for (int i=0; i<nsubI*nsubJ; ++i) {
        dvSubPtr[i] = dvs;
    }
}


}
