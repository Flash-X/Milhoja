#ifndef TILEITER_H__
#define TILEITER_H__

#include "Tile.h"

namespace orchestration {

class TileIter {
public:
    // TODO: implement tiling!!
    TileIter() {}

    TileIter(TileIter&&) = default;
    virtual ~TileIter(void) {}

    virtual bool isValid() const  = 0;
    virtual void next() = 0;
    virtual std::unique_ptr<Tile> buildCurrentTile() = 0;

private:
    TileIter(const TileIter&) = delete;
    TileIter& operator=(TileIter&&) = delete;
    // Limit all copies as much as possible
    TileIter(TileIter&) = delete;
    TileIter& operator=(TileIter&) = delete;
    TileIter& operator=(const TileIter&) = delete;
};

}

#endif

