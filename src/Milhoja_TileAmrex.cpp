#include "Milhoja_TileAmrex.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>

#include "Milhoja_Logger.h"
#include "Milhoja_Grid.h"

namespace milhoja {

/**
 * \brief Constructor for TileAmrex
 *
 * Should be called from inside a Tile Iterator, specifically:
 * TileIterAmrex::buildCurrentTile. Initializes private members.
 *
 * \param itor An AMReX MFIter currently iterating.
 * \param unkRef A ref to the multifab being iterated over.
 * \param level Level of iterator.
 */
TileAmrex::TileAmrex(amrex::MFIter& itor, amrex::MultiFab& unkRef, const unsigned int level)
    : Tile{},
      unkRef_{unkRef},
      level_{level},
      gridIdx_{ itor.index() },
      interior_{ new amrex::Box(itor.validbox()) }, //TODO tiling?
      GC_{ new amrex::Box(itor.fabbox()) }          //TODO tiling?
{
    amrex::FArrayBox& fab = unkRef_[gridIdx_];

#ifdef DEBUG_RUNTIME
    std::string   msg = "[TileAmrex] Created Tile object "
                  + std::to_string(gridIdx_)
                  + " from MFIter";
    Logger::instance().log(msg);
#endif
}

/**
 * \brief Destructor for TileAmrex
 *
 * Deletes/nullifies private members.
 */
TileAmrex::~TileAmrex(void) {
    if (interior_) {
        delete interior_;
        interior_ = nullptr;
    }
    if (GC_) {
        delete GC_;
        GC_ = nullptr;
    }
#ifdef DEBUG_RUNTIME
    std::string msg = "[TileAmrex] Destroying Tile object "
                      + std::to_string(gridIdx_);
    Logger::instance().log(msg);
#endif
}

/**
 * \brief Checks whether a Tile is null.
 */
bool   TileAmrex::isNull(void) const {
    return (   (gridIdx_ < 0) //TODO this is never true?
            && (level_ == 0) 
            && (interior_    == nullptr)
            && (GC_          == nullptr));
}

/**
 * \brief Gets index of lo cell in the Tile
 *
 * \return IntVect with index of lower left cell.
 */
IntVect  TileAmrex::lo(void) const {
    return IntVect(interior_->smallEnd());
}

/**
 * \brief Gets index of hi cell in the Tile
 *
 * \return IntVect with index of upper right cell.
 */
IntVect  TileAmrex::hi(void) const {
    return IntVect(interior_->bigEnd());
}


/**
 * \brief Gets index of lo guard cell in the Tile
 *
 * \return IntVect with index of lower left cell, including
 *         guard cells.
 */
IntVect  TileAmrex::loGC(void) const {
    return IntVect(GC_->smallEnd());
}

/**
 * \brief Gets index of hi guard cell in the Tile
 *
 * \return IntVect with index of upper right cell, including
 *         guard cells.
 */
IntVect  TileAmrex::hiGC(void) const {
    return IntVect(GC_->bigEnd());
}

/**
 * \brief Returns pointer to underlying data structure.
 *
 * \return Real* pointing to underlying data.
 */
Real*   TileAmrex::dataPtr(void) {
    //TODO cache the ptr? (eager vs lazy)
    return static_cast<Real*>(unkRef_[gridIdx_].dataPtr());
}

/**
 * \brief Returns FArray4D to access underlying data.
 *
 * \return A FArray4D object which wraps the pointer to underlying
 *         data and provides Fortran-style access.
 */
FArray4D TileAmrex::data(void) {
    int            nComp   = unkRef_.nComp();
    assert(nComp >= 0);
    unsigned int   nCcVars = static_cast<unsigned int>(nComp);

    return FArray4D{dataPtr(), loGC(), hiGC(), nCcVars};
}

}

