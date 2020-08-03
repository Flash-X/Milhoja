#include "TileAmrex.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>

#include "Grid.h"
#include "Flash.h"
#include "constants.h"

namespace orchestration {

/**
 *
 */
TileAmrex::TileAmrex(amrex::MFIter& itor, const unsigned int level)
    : Tile(level),
      unk_(Grid::instance().unk())
{
    gridIdx_ = itor.index();
    interior_ = new amrex::Box(itor.tilebox());
    GC_ = new amrex::Box(itor.fabbox());
    amrex::FArrayBox& fab = unk_[gridIdx_];
    CC_h_ = fab.dataPtr();
}

TileAmrex::TileAmrex(TileAmrex&& other)
    : Tile( std::move(other) ),
      unk_{other.unk_}
{
}

TileAmrex& TileAmrex::operator=(TileAmrex&& rhs)
{
    Tile::operator=(std::move(rhs));
    return *this;
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
    return static_cast<Real*>(unk_[gridIdx_].dataPtr()); 
}

/**
 *
 */
FArray4D TileAmrex::data(void) {
    return FArray4D{static_cast<Real*>(unk_[gridIdx_].dataPtr()),
                    loGC().asTriple(), hiGC().asTriple(), NUNKVAR}; 
}

}

