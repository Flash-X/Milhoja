#include "Tile.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>

#include "Grid.h"
#include "OrchestrationLogger.h"
#include "Flash.h"
#include "constants.h"

namespace orchestration {

/**
 *
 */
Tile::Tile(amrex::MFIter& itor, const unsigned int level)
    : DataItem{},
      CC_h_{nullptr},
      CC1_p_{nullptr},
      CC2_p_{nullptr},
      loGC_p_{nullptr},
      hiGC_p_{nullptr},
      CC1_d_{nullptr},
      CC2_d_{nullptr},
      loGC_d_{nullptr},
      hiGC_d_{nullptr},
      CC1_array_d_{nullptr},
      gridIdx_{itor.index()},
      level_{0},
      interior_{new amrex::Box(itor.tilebox())},
      GC_{new amrex::Box(itor.fabbox())}
{
    amrex::MultiFab&  unk = Grid::instance().unk();
    amrex::FArrayBox& fab = unk[gridIdx_];
    CC_h_ = fab.dataPtr();

#ifdef DEBUG_RUNTIME
    std::string   msg =   "[Tile] Created Tile object "
                        + std::to_string(gridIdx_)
                        + " from MFIter";
    Logger::instance().log(msg);
#endif
}

Tile::Tile(Tile&& other)
    : DataItem{},
      CC_h_{other.CC_h_},
      CC1_p_{other.CC1_p_},
      CC2_p_{other.CC2_p_},
      loGC_p_{other.loGC_p_},
      hiGC_p_{other.hiGC_p_},
      CC1_d_{other.CC1_d_},
      CC2_d_{other.CC2_d_},
      loGC_d_{other.loGC_d_},
      hiGC_d_{other.hiGC_d_},
      CC1_array_d_{other.CC1_array_d_},
      gridIdx_{other.gridIdx_},
      level_{other.level_},
      interior_{other.interior_},
      GC_{other.GC_}
{
    // The assumption here is that interior_/GC_ were allocated dynamically
    // beforehand and by moving the pointers to this object, it is this object's
    // responsibility deallocate the associated resources.
    other.CC_h_        = nullptr;
    other.CC1_p_       = nullptr;
    other.CC2_p_       = nullptr;
    other.loGC_p_      = nullptr;
    other.hiGC_p_      = nullptr;
    other.CC1_d_       = nullptr;
    other.CC2_d_       = nullptr;
    other.loGC_d_      = nullptr;
    other.hiGC_d_      = nullptr;
    other.CC1_array_d_ = nullptr;
    other.gridIdx_     = -1;
    other.level_       = 0;
    other.interior_    = nullptr;
    other.GC_          = nullptr;

#ifdef DEBUG_RUNTIME
    std::string msg =    "[Tile] Moved Tile object "
                      +  std::to_string(gridIdx_) +  " by move constructor";
    Logger::instance().log(msg);
#endif
}

Tile& Tile::operator=(Tile&& rhs) {
    CC_h_        = rhs.CC_h_;
    CC1_p_       = rhs.CC1_p_;
    CC2_p_       = rhs.CC2_p_;
    loGC_p_      = rhs.loGC_p_;
    hiGC_p_      = rhs.hiGC_p_;
    CC1_d_       = rhs.CC1_d_;
    CC2_d_       = rhs.CC2_d_;
    loGC_d_      = rhs.loGC_d_;
    hiGC_d_      = rhs.hiGC_d_;
    CC1_array_d_ = rhs.CC1_array_d_;
    gridIdx_     = rhs.gridIdx_;
    level_       = rhs.level_;
    interior_    = rhs.interior_;
    GC_          = rhs.GC_;

    // The assumption here is that interior_/GC_ were allocated dynamically
    // beforehand and by moving the pointers to this object, it is this object's
    // responsibility deallocate the associated resources.
    rhs.CC_h_        = nullptr;
    rhs.CC1_p_       = nullptr;
    rhs.CC2_p_       = nullptr;
    rhs.loGC_p_      = nullptr;
    rhs.hiGC_p_      = nullptr;
    rhs.CC1_d_       = nullptr;
    rhs.CC2_d_       = nullptr;
    rhs.loGC_d_      = nullptr;
    rhs.hiGC_d_      = nullptr;
    rhs.CC1_array_d_ = nullptr;
    rhs.gridIdx_     = -1;
    rhs.level_       = 0;
    rhs.interior_    = nullptr;
    rhs.GC_          = nullptr;
#ifdef DEBUG_RUNTIME
    std::string msg =   "[Tile] Moved Tile object "
                      + std::to_string(gridIdx_) + " by move assignment";
    Logger::instance().log(msg);
#endif

    return *this;
}

/**
 *
 */
Tile::~Tile(void) {
#ifdef DEBUG_RUNTIME
    std::string msg =   "[Tile] Destroying Tile object " + std::to_string(gridIdx_);
    Logger::instance().log(msg);
#endif

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
bool   Tile::isNull(void) const {
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
IntVect  Tile::lo(void) const {
    return IntVect(interior_->smallEnd());
}

/**
 *
 */
IntVect  Tile::hi(void) const {
    return IntVect(interior_->bigEnd());
}

/**
 *
 */
const amrex::Box&  Tile::interior(void) const {
    return (*interior_);
}

/**
 *
 */
amrex::Dim3  Tile::loGC(void) const {
    return amrex::lbound(*GC_);
}

/**
 *
 */
amrex::Dim3  Tile::hiGC(void) const {
    return amrex::ubound(*GC_);
}

/**
 *
 */
const int*  Tile::loGCVect(void) const {
    return GC_->loVect();
}

/**
 *
 */
const int*  Tile::hiGCVect(void) const {
    return GC_->hiVect();
}

/**
 *
 */
const amrex::Box&  Tile::interiorAndGC(void) const {
    return (*GC_);
}

/**
 *
 */
amrex::XDim3 Tile::deltas(void) const {
    amrex::Geometry& geometry = Grid::instance().geometry();

    amrex::XDim3   deltas;

    deltas.x = geometry.CellSize(0);
    deltas.y = geometry.CellSize(1);
    deltas.z = geometry.CellSize(2);

    return deltas;
}

/**
 *
 */
std::size_t Tile::nSubItems(void) const {
    throw std::logic_error("[Tile::nSubItems] Tiles do not have sub items");
}

/**
 *
 */
std::shared_ptr<DataItem>  Tile::popSubItem(void) {
    throw std::logic_error("[Tile::popSubItem] Tiles do not have sub items");
}

/**
 *
 */
DataItem*  Tile::getSubItem(const std::size_t i) {
    throw std::logic_error("[Tile::getSubItem] Tiles do not have sub items");
}

/**
 *
 */
void  Tile::addSubItem(std::shared_ptr<DataItem>&& dataItem) {
    throw std::logic_error("[Tile::addSubItem] Tiles do not have sub items");
}

}
