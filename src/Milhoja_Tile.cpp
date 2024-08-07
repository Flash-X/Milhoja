#include "Milhoja_Tile.h"

#include "Milhoja_Grid.h"
#include "Milhoja_Logger.h"

namespace milhoja {

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


#ifdef RUNTIME_MUST_USE_TILEITER
/**
 * \brief Get deltas for appropriate level.
 */
RealVect Tile::deltas(void) const {
    return Grid::instance().getDeltas(level());
}
#endif

#ifdef FULL_MILHOJAGRID
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
#endif

}
