#include "Tile.h"

#include "Grid.h"
#include "OrchestrationLogger.h"

namespace orchestration {

/**
 * \brief Default constructor for Tile
 *
 * All pointers start as null.
 */
Tile::Tile(void)
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
      CC1_array_d_{nullptr} {
}

/**
 * \brief Move constructor for Tile
 */
/*Tile::Tile(Tile&& other)
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
      CC1_array_d_{other.CC1_array_d_}
{
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

#ifdef DEBUG_RUNTIME
    std::string msg =    "[Tile] Moved Tile object by move constructor";
    Logger::instance().log(msg);
#endif
}*/

/**
 * \brief Move assignment operator for Tile
 */
/*Tile& Tile::operator=(Tile&& rhs) {
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
#ifdef DEBUG_RUNTIME
    std::string msg =   "[Tile] Moved Tile object by move assignment";
    Logger::instance().log(msg);
#endif

    return *this;
}*/

/**
 * \brief Detructor for Tile
 *
 * Resets all pointers to null.
 */
Tile::~Tile(void) {
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
#ifdef DEBUG_RUNTIME
    std::string msg =   "[Tile] Destroying Tile object";
    Logger::instance().log(msg);
#endif
}


/**
 * \brief Get deltas for appropriate level.
 */
RealVect Tile::deltas(void) const {
    return Grid::instance().getDeltas(level());
}

/**
  * \brief Get the physical coordinates of the
  *        center of the tile.
  *
  * @return A real vector with the physical center coordinates of the tile.
  */
RealVect Tile::getCenterCoords(void) const {
    Grid& grid = Grid::instance();
    RealVect dx = deltas();
    RealVect x0 = grid.getProbLo();
    IntVect offset = grid.getDomainLo(level());
    IntVect loPt = lo() - offset;
    IntVect hiPt = hi() - offset + 1;
    RealVect coords = x0 + dx*RealVect(loPt+hiPt)*0.5_wp;
    return coords;
}


/**
 *
 */
void Tile::unpack(void) {
    throw std::logic_error("[Tile::unpack] Tiles are not packed");
}

/**
 *
 */
void* Tile::hostPointer(void) {
    throw std::logic_error("[Tile::hostPointer] No need for host pointer");
}

/**
 *
 */
void* Tile::gpuPointer(void) {
    throw std::logic_error("[Tile::gpuPointer] No need for gpu pointer");
}

/**
 *
 */
std::size_t  Tile::sizeInBytes(void) {
    throw std::logic_error("[Tile::sizeInBytes] Don't need tile size");
}

/**
 *
 */
std::shared_ptr<DataItem> Tile::getTile(void) {
    throw std::logic_error("[Tile::getTile] Not needed");
}

/**
 *
 */
CudaStream&   Tile::stream(void) {
    throw std::logic_error("[Tile::stream] Not needed");
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

/**
 *
 */
void  Tile::transferFromDeviceToHost(void) {
    throw std::logic_error("[Tile::transferFromDeviceToHost] Tiles cannot be transferred");
}

}
