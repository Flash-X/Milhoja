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
    : DataItem{}
{ }

/**
 * \brief Detructor for Tile
 *
 * Resets all pointers to null.
 */
Tile::~Tile(void) {
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
