#ifndef MILHOJA_TILE_ITER_H__
#define MILHOJA_TILE_ITER_H__

#include <memory>

namespace milhoja {

class Tile;

/**
  * \brief Class for iterating over data in a level.
  *
  * TileIter is a pure abstract class. A derived class implementing the
  * virtual member functions should be written for each AMR package.
  *
  * Use in for-loops:
  *   `for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next())`
  *
  * \todo Implement tiling
  * \todo Consider how to make multi-level iterators
  */
class TileIter {
public:
    TileIter() {}
    virtual ~TileIter(void) = default;

    TileIter(TileIter&&) = delete;
    TileIter(const TileIter&) = delete;
    TileIter& operator=(TileIter&&) = delete;
    TileIter(TileIter&) = delete;
    TileIter& operator=(TileIter&) = delete;
    TileIter& operator=(const TileIter&) = delete;

    // Pure virtual functions.
    virtual bool isValid(void) const  = 0;
    virtual void next(void) = 0;
    virtual std::unique_ptr<Tile> buildCurrentTile(void) = 0;
    virtual Tile*                 buildCurrentTile_forFortran(void) = 0;

private:
};

}

#endif

