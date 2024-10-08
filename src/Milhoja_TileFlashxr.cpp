#ifdef DEBUG_RUNTIME
#include <string>
#endif

#include "Milhoja.h"
#include "Milhoja_TileFlashxr.h"

#include "Milhoja_Logger.h"

#include <cassert>


namespace milhoja {

/**
 * \brief Constructor for TileFlashxr
 *
 * Should be constructed from C- and Fortran-compatible data.
 * Initializes private members.
 *
 * \todo Include a single metadata routine that gets gId, level, lo/hi, and
 * loGC/hiGC in one call?  This could replace the lo(int*), etc. calls.
 *
 * \note Refer Milhoja_FlashxrTileRaw.h
 *
 * \param tP      The struct that has raw pointers for constructing Flash-X tileDesc
 *                e.g.) Unk, FLUX[XYZ]
 * \param tI      The struct that has integer variables for constructing Flash-X tileDesc
 *                e.g.) level, tileIdx, ...
 * \param tR      The struct that has real variables for constructing Flash-X tileDesc
 *                e.g.) delta[XYZ]
 */
TileFlashxr::TileFlashxr(const FlashxrTileRawPtrs tP,
                         const FlashxTileRawInts tI,
                         const FlashxTileRawReals tR)
    : Tile{},
      level_{static_cast<unsigned int>(tI.level - 1)},     // 1-based Flash-X convention -> 0-based
      gridIdxOrBlkId_{tI.gridIdxOrBlkId},
      tileIdx_{tI.tileIdx},
      lo_{LIST_NDIM(tI.loX-1,tI.loY-1,tI.loZ-1)},          // 1-based Flash-X convention -> 0-based
      hi_{LIST_NDIM(tI.hiX-1,tI.hiY-1,tI.hiZ-1)},          // 1-based Flash-X convention -> 0-based
      loGC_{LIST_NDIM(tI.loGCX-1,tI.loGCY-1,tI.loGCZ-1)},  // 1-based Flash-X convention -> 0-based
      hiGC_{LIST_NDIM(tI.hiGCX-1,tI.hiGCY-1,tI.hiGCZ-1)},  // 1-based Flash-X convention -> 0-based
      nCcComp_{tI.nCcComp},
      nFluxComp_{tI.nFluxComp},
      deltas_{LIST_NDIM(tR.deltaX,tR.deltaY,tR.deltaZ)},
      unkBlkPtr_{tP.unkBlkPtr},
      fluxBlkPtrs_{std::vector<Real*>{nullptr, nullptr, nullptr}}
{
    fluxBlkPtrs_[0] = tP.fluxxBlkPtr;
#if MILHOJA_NDIM > 1
    fluxBlkPtrs_[1] = tP.fluxyBlkPtr;
#endif
#if MILHOJA_NDIM > 2
    fluxBlkPtrs_[2] = tP.fluxzBlkPtr;
#endif

#ifdef DEBUG_RUNTIME
    std::string msg =   "[TileFlashxr] Created Tile (level="
                      + std::to_string(level_)
                      + " / grid ID=" + std::to_string(gridIdxOrBlkId_)
                      + " / tile ID=" + std::to_string(tileIdx_) + ") with "
                      + std::to_string(fluxBlkPtrs_.size()) + " flux vars";
    Logger::instance().log(msg);
#endif
}

/**
 * \brief Destructor for TileFlashxr
 *
 * Deletes/nullifies private members.
 */
TileFlashxr::~TileFlashxr(void) {
#ifdef DEBUG_RUNTIME
    std::string msg =   "[TileFlashxr] Destroyed Tile (level="
                      + std::to_string(level_)
                      + " / grid ID=" + std::to_string(gridIdxOrBlkId_)
                      + " / tile ID=" + std::to_string(tileIdx_)
                      + ")";
    Logger::instance().log(msg);
#endif
}

unsigned int    TileFlashxr::nCcVariables(void) const {
    int    nCcVars{nCcComp_};
    assert(nCcVars >= 0);
    return static_cast<unsigned int>(nCcVars);
}

unsigned int    TileFlashxr::nFluxVariables(void) const {
    if (fluxBlkPtrs_.size() == 0) {
        return 0;
    }

    int    nFluxVars{nFluxComp_};
    assert(nFluxVars >= 0);
    return static_cast<unsigned int>(nFluxVars);
}

/**
 * \brief Gets physical cell size (coordinate delta) for each cell in the Tile
 *
 * \return RealVect with index of lower left cell.
 */
RealVect  TileFlashxr::deltas(void) const {
  return RealVect{LIST_NDIM(deltas_[0],deltas_[1],deltas_[2])};
  //  return deltas_;
}

/**
 * \brief Gets index of lo cell in the Tile
 *
 * \return IntVect with index of lower left cell.
 */
IntVect  TileFlashxr::lo(void) const {
    return IntVect{LIST_NDIM(lo_[0],lo_[1],lo_[2])};
}

/**
 * \brief Gets index of hi cell in the Tile
 *
 * \return IntVect with index of upper right cell.
 */
IntVect  TileFlashxr::hi(void) const {
    return IntVect{LIST_NDIM(hi_[0],hi_[1],hi_[2])};
}

/**
 * \brief Gets index of lo guard cell in the Tile
 *
 * \return IntVect with index of lower left cell, including
 *         guard cells.
 */
IntVect  TileFlashxr::loGC(void) const {
  return IntVect{LIST_NDIM(loGC_[0],loGC_[1],loGC_[2])};
}

/**
 * \brief Gets index of hi guard cell in the Tile
 *
 * \return IntVect with index of upper right cell, including
 *         guard cells.
 */
IntVect  TileFlashxr::hiGC(void) const {
  return IntVect{LIST_NDIM(hiGC_[0],hiGC_[1],hiGC_[2])};
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
Real*   TileFlashxr::dataPtr(void) {
    return unkBlkPtr_;
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
std::vector<Real*>   TileFlashxr::fluxDataPtrs(void) {
    std::vector<Real*>   fluxPtrs{fluxBlkPtrs_.size(), nullptr};
    for (auto i=0; i<fluxBlkPtrs_.size(); ++i) {
        fluxPtrs[i] = static_cast<Real*>(fluxBlkPtrs_[i]);
    }
    return fluxPtrs;
}

/**
 * \brief Returns FArray4D to access underlying data.
 *
 * \return A FArray4D object which wraps the pointer to underlying
 *         data and provides Fortran-style access.
 */
FArray4D TileFlashxr::data(void) {
    int   nCcVars = nCcComp_;
    assert(nCcVars >= 0);
    return FArray4D{dataPtr(), loGC(), hiGC(), static_cast<unsigned int>(nCcVars)};
}

/**
 * \brief Returns FArray4D to access underlying data.
 *
 * \return A FArray4D object which wraps the pointer to underlying
 *         data and provides Fortran-style access.
 */
FArray4D TileFlashxr::fluxData(const unsigned int dir) {
    if (fluxBlkPtrs_.size() == 0) {
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

    assert(fluxBlkPtrs_[dir]);
    int     nFluxVars_signed = nFluxComp_;
    Real*   dataPtr = fluxBlkPtrs_[dir];

    assert(nFluxVars_signed >= 0);
    unsigned int nFluxVars = static_cast<unsigned int>(nFluxVars_signed);

    IntVect    loFlux = lo();
    IntVect    hiFlux = hi();
    hiFlux[dir] += 1;

    return FArray4D{dataPtr, loFlux, hiFlux, nFluxVars};
}

}

