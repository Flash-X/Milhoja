#include "Grid.h"

#include <stdexcept>
#include "Grid_Axis.h"
#include "Grid_Edge.h"

// TODO: move to a header?
#ifdef GRID_AMREX
#include "GridAmrex.h"
namespace orchestration {
    typedef GridAmrex GridVersion;
}
#else
throw std::logic_error("Need to specify Grid implementation with GRID_[NAME] macro.");
#endif


namespace orchestration {


/**
 * instace gets a reference to the singleton Grid object.
 *
 * @return A reference to the singleton object, which has been downcast to Grid type.
 */
Grid&   Grid::instance(void) {
    if(!instantiated_) {
        throw std::logic_error("Cannot call Grid::instance until after Grid::instantiate has been called.");
    }
    static GridVersion gridSingleton;
    return gridSingleton;
}

/**
 * instantiate allows the user to easily distinguish the first call to instance(), which
 * calls the Grid constructor, from all subsequent calls. It must be called exactly once in
 * the program, before all calls to instance().
 */
void   Grid::instantiate(void) {
    if(instantiated_) {
        throw std::logic_error("Cannot call Grid::instantiate after Grid has already been initialized.");
    }
    instantiated_ = true;
    Grid::instance();
}

/**
  * getDeltas gets the cell size for a given level.
  *
  * @param level The level of refinement (0 is coarsest).
  * @return The vector <dx,dy,dz> for a given level.
  */
RealVect    Grid::getDeltas(const unsigned int level) const {
    throw std::logic_error("Grid::getDeltas not implemented");
    return RealVect{LIST_NDIM(0.0_wp,0.0_wp,0.0_wp)};
}

/**
  * getBlkCenterCoords gets the physical coordinates of the
  * center of the given tile.
  *
  * @param tileDesc A Tile object.
  * @return A real vector with the physical center coordinates of the tile.
  */
RealVect    Grid::getBlkCenterCoords(const Tile& tileDesc) const {
    RealVect dx = getDeltas(tileDesc.level());
    RealVect x0 = getProbLo();
    IntVect lo = tileDesc.lo();
    IntVect hi = tileDesc.hi();
    RealVect coords = x0 + dx*RealVect(lo+hi+1)*0.5_wp;
    return coords;
}

/** getCellFaceAreaLo gets lo face area of a cell with given (integer) coordinates
  *
  * @param axis Axis of desired face, returns the area of the lo side.
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return area of face (Real)
  */
Real  Grid::getCellFaceAreaLo(const unsigned int axis, const unsigned int lev, const IntVect& coord) const {
    throw std::logic_error("Grid::getCellFaceAreaLo not implemented");
    return 0.0_wp;
}

/** getCellVolume gets the volume of a cell with given (integer) coordinates
  *
  * @param lev Level (0-based)
  * @param coord Cell-centered coordinates (integer, 0-based)
  * @return Volume of cell (Real)
  */
Real  Grid::getCellVolume(const unsigned int lev, const IntVect& coord) const {
    throw std::logic_error("Grid::getCellVolume not implemented");
    return 0.0_wp;
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
  */
void    Grid::fillCellCoords(const unsigned int axis, const unsigned int edge, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* coordPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("Grid::fillCellCoords: Invalid axis.");
    }
    if(edge!=Edge::Left && edge!=Edge::Right && edge!=Edge::Center){
        throw std::logic_error("Grid::fillCellCoords: Invalid edge.");
    }
#endif
    throw std::logic_error("Grid::fillCellCoords not implemented");
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
  */
void    Grid::fillCellFaceAreasLo(const unsigned int axis, const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* areaPtr) const {
#ifndef GRID_ERRCHECK_OFF
    if(axis!=Axis::I && axis!=Axis::J && axis!=Axis::K ){
        throw std::logic_error("Grid::fillCellFaceAreasLo: Invalid axis.");
    }
#endif
    throw std::logic_error("Grid::fillCellFaceAreasLo not implemented");
}


/** fillCellVolumes fills a Real array (passed by pointer) with the
  * volumes of cells in a given range
  *
  * @param lev Level (0-based)
  * @param lo Lower bound of range (cell-centered 0-based integer coordinates)
  * @param hi Upper bound of range (cell-centered 0-based integer coordinates)
  * @param vols Real Ptr to some fortran-style data structure. Will be filled with volumes.
  *             Should be of shape (lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2], 1).
  */
void    Grid::fillCellVolumes(const unsigned int lev, const IntVect& lo, const IntVect& hi, Real* volPtr) const {
    throw std::logic_error("Grid::fillCellVolumes not implemented");
}


bool Grid::instantiated_ = false;


}
