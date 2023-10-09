#ifdef DEBUG_RUNTIME
#include <string>
#endif

#include "Milhoja.h"
#include "Milhoja_TileAmrex.h"

#include "Milhoja_Logger.h"
#include "Milhoja_Grid.h"

namespace milhoja {

/**
 * \brief Constructor for TileAmrex
 *
 * Should be called from inside a Tile Iterator, specifically:
 * TileIterAmrex::buildCurrentTile. Initializes private members.
 *
 * \todo Include a single metadata routine that gets gId, level, lo/hi, and
 * loGC/hiGC in one call?  This could replace the lo(int*), etc. calls.
 *
 * \param level      The 0-based refinement level of the tile
 * \param gridIdx    The integer index for the block that contains the tile
 * \param tileIdx    The local integer index for the tile in its block
 * \param interior   The lo/hi global indices of the tile's interior cells
 * \param dataArray  The lo/hi global indices of the tile's cell-centered data array
 * \param unkFab     The FAB containing the cell-centered data at the given
 *                   level
 * \param fluxFabs   A vector of pointers to the FABs that store the
 *                   face-centered flux data at the given level.  If there are
 *                   no flux variables for the problem, this vector should be
 *                   empty.  Otherwise, it should have MILHOJA_NDIM elements.
 */
TileAmrex::TileAmrex(const unsigned int level,
                     const int gridIdx,
                     const int tileIdx,
                     const amrex::Box&& interior,
                     const amrex::Box&& dataArray,
                     amrex::FArrayBox& unkFab,
                     std::vector<amrex::FArrayBox*>&& fluxFabs)
    : Tile{},
      level_{level},
      gridIdx_{gridIdx},
      tileIdx_{tileIdx},
      interiorBox_{std::move(interior)},
      dataArrayBox_{std::move(dataArray)},
      unkFab_{unkFab},
      fluxFabs_{std::move(fluxFabs)}
{
    for (auto i=0; i<fluxFabs_.size(); ++i) {
        if (!(fluxFabs_[i])) {
            throw std::invalid_argument("[TileAmrex::TileAmrex] Null flux FAB pointer");
        }
    }

#ifdef DEBUG_RUNTIME
    std::string msg =   "[TileAmrex] Created Tile (level="
                      + std::to_string(level_)
                      + " / grid ID=" + std::to_string(gridIdx_)
                      + " / tile ID=" + std::to_string(tileIdx_) + ") with "
                      + std::to_string(fluxFabs_.size()) + " flux FABs";
    Logger::instance().log(msg);
#endif
}

/**
 * \brief Destructor for TileAmrex
 *
 * Deletes/nullifies private members.
 */
TileAmrex::~TileAmrex(void) {
#ifdef DEBUG_RUNTIME
    std::string msg =   "[TileAmrex] Destroyed Tile (level="
                      + std::to_string(level_)
                      + " / grid ID=" + std::to_string(gridIdx_)
                      + " / tile ID=" + std::to_string(tileIdx_)
                      + ")";
    Logger::instance().log(msg);
#endif
}

unsigned int    TileAmrex::nCcVariables(void) const {
    int    nCcVars{unkFab_.nComp()};
    assert(nCcVars >= 0);
    return static_cast<unsigned int>(nCcVars);
}

unsigned int    TileAmrex::nFluxVariables(void) const {
    if (fluxFabs_.size() == 0) {
        return 0;
    }

    int    nFluxVars{fluxFabs_[0]->nComp()};
    assert(nFluxVars >= 0);
    return static_cast<unsigned int>(nFluxVars);
}

/**
 * \brief Gets index of lo cell in the Tile
 *
 * \return IntVect with index of lower left cell.
 */
IntVect  TileAmrex::lo(void) const {
    return IntVect(interiorBox_.smallEnd());
}

/**
 * \brief Gets index of hi cell in the Tile
 *
 * \return IntVect with index of upper right cell.
 */
IntVect  TileAmrex::hi(void) const {
    return IntVect(interiorBox_.bigEnd());
}

/**
 * \brief Gets index of lo guard cell in the Tile
 *
 * \return IntVect with index of lower left cell, including
 *         guard cells.
 */
IntVect  TileAmrex::loGC(void) const {
    return IntVect(dataArrayBox_.smallEnd());
}

/**
 * \brief Gets index of hi guard cell in the Tile
 *
 * \return IntVect with index of upper right cell, including
 *         guard cells.
 */
IntVect  TileAmrex::hiGC(void) const {
    return IntVect(dataArrayBox_.bigEnd());
}

/**
 * \brief Returns pointer to underlying data structure.
 *
 * \todo This routine should return the lo/hi and shape of the data associated
 *       with the pointer.  AMReX dictates what we point to and this is
 *       analogous to wrapping the data with FArray4D.
 *
 * \return Pointer to start of tile's data in host memory.
 */
Real*   TileAmrex::dataPtr(void) {
    return static_cast<Real*>(unkFab_.dataPtr());
}

/**
 * \brief Returns pointer to underlying data structure.
 *
 * \todo This routine should return the lo/hi and shape of the data associated
 *       with the pointer.  AMReX dictates what we point to and this is
 *       analogous to wrapping the data with FArray4D.
 *
 * \return Pointer to start of tile's data in host memory.
 */
std::vector<Real*>   TileAmrex::fluxDataPtrs(void) {
    std::vector<Real*>   fluxPtrs{fluxFabs_.size(), nullptr};
    for (auto i=0; i<fluxFabs_.size(); ++i) {
        fluxPtrs[i] = static_cast<Real*>(fluxFabs_[i]->dataPtr());
    }
    return fluxPtrs;
}

/**
 * \brief Returns FArray4D to access underlying data.
 *
 * \return A FArray4D object which wraps the pointer to underlying
 *         data and provides Fortran-style access.
 */
FArray4D TileAmrex::data(void) {
    int   nCcVars = unkFab_.nComp();
    assert(nCcVars >= 0);
    return FArray4D{dataPtr(), loGC(), hiGC(), static_cast<unsigned int>(nCcVars)};
}

/**
 * \brief Returns FArray4D to access underlying data.
 *
 * \return A FArray4D object which wraps the pointer to underlying
 *         data and provides Fortran-style access.
 */
FArray4D TileAmrex::fluxData(const unsigned int dir) {
    if (fluxFabs_.size() == 0) {
        throw std::logic_error("No flux data available");
    }

#if MILHOJA_NDIM == 1
    if (dir == Axis::J) {
        throw std::logic_error("No J-axis flux for 1D problem");
    }
#endif
#if MILHOJA_NDIM <= 2
    if (dir == Axis::K) {
        throw std::logic_error("No K-axis flux for problem");
    }
#endif

    assert(fluxFabs_[dir]);
    int     nFluxVars_amrex = fluxFabs_[dir]->nComp();
    Real*   dataPtr = static_cast<Real*>(fluxFabs_[dir]->dataPtr());

    assert(nFluxVars_amrex >= 0);
    unsigned int nFluxVars = static_cast<unsigned int>(nFluxVars_amrex);

    IntVect    loFlux = lo();
    IntVect    hiFlux = hi();
    hiFlux[dir] += 1;

    return FArray4D{dataPtr, loFlux, hiFlux, nFluxVars};
}

}

