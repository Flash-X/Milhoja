#include "TileAmrex.h"

#include "Grid.h"
#include "OrchestrationLogger.h"
#include "Flash.h"
#include "constants.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>

namespace orchestration {

/**
 *
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
    CC_h_ = fab.dataPtr();

#ifdef DEBUG_RUNTIME
    std::string   msg =   "[Tile] Created Tile object "
                  + std::to_string(gridIdx_)
                  + " from MFIter";
    Logger::instance().log(msg);
#endif
}

/**
 *
 */
TileAmrex::~TileAmrex(void) {
    CC_h_        = nullptr;
    CC1_p_       = nullptr;
    CC2_p_       = nullptr;
    loGC_p_      = nullptr;
    hiGC_p_      = nullptr;
    CC1_d_       = nullptr;
    CC2_d_       = nullptr;
    loGC_d_      = nullptr;
    hiGC_d_      = nullptr;
    CC1_array_d_ = nullptr;

    if (interior_) {
        delete interior_;
        interior_ = nullptr;
    }
    if (GC_) {
        delete GC_;
        GC_ = nullptr;
    }
#ifdef DEBUG_RUNTIME
    std::string msg =   "[Tile] Destroying Tile object " + std::to_string(gridIdx_);
    Logger::instance().log(msg);
#endif
}

/**
 *
 */
bool   TileAmrex::isNull(void) const {
    return (   (gridIdx_ < 0)
            && (level_ == 0) 
            && (interior_    == nullptr)
            && (GC_          == nullptr)
            && (CC_h_        == nullptr)
            && (CC1_p_       == nullptr)
            && (CC2_p_       == nullptr)
            && (loGC_p_      == nullptr)
            && (hiGC_p_      == nullptr)
            && (CC1_d_       == nullptr)
            && (CC2_d_       == nullptr)
            && (loGC_d_      == nullptr)
            && (hiGC_d_      == nullptr)
            && (CC1_array_d_ == nullptr));
}
/**
 *
 */
IntVect  TileAmrex::lo(void) const {
    return IntVect(interior_->smallEnd());
}

/**
 *
 */
IntVect  TileAmrex::hi(void) const {
    return IntVect(interior_->bigEnd());
}


/**
 *
 */
IntVect  TileAmrex::loGC(void) const {
    return IntVect(GC_->smallEnd());
}

/**
 *
 */
IntVect  TileAmrex::hiGC(void) const {
    return IntVect(GC_->bigEnd());
}

/**
 *
 */
Real*   TileAmrex::dataPtr(void) {
    // TODO use CC_h_? Then don't have to store unkRef
    return static_cast<Real*>(unkRef_[gridIdx_].dataPtr());
}

/**
 *
 */
FArray4D TileAmrex::data(void) {
    // TODO use CC_h_?
    return FArray4D{static_cast<Real*>(unkRef_[gridIdx_].dataPtr()),
                    loGC().asTriple(), hiGC().asTriple(), NUNKVAR}; 
}

}

