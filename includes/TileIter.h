#ifndef TILEITER_H__
#define TILEITER_H__

#include <memory>

namespace orchestration {

class Tile;

/**
  * \brief Class for iterating over data in a level.
  *
  * TileIter is a pure abstract class. A derived class implementing the
  * virtual member functions should be written for each AMR package.
  *
  * Use in for-loops:
  *   `for (auto ti = grid.buildTileIter(0); ti->isValid(); ti->next())`
  *
  * TODO: implement tiling!!
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
    virtual bool isValid() const  = 0;
    virtual void next() = 0;
    virtual std::unique_ptr<Tile> buildCurrentTile() = 0;

private:
};

}

#endif

