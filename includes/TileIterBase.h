#ifndef TILEITERBASE_H__
#define TILEITERBASE_H__

#include "Tile.h"

namespace orchestration {

class TileIterBase {
public:
    // TODO: implement tiling!!
    TileIterBase() {}

    TileIterBase(TileIterBase&&) = default;
    virtual ~TileIterBase(void) {}

    virtual bool isValid() const  = 0;
    virtual void operator++() = 0;
    virtual std::unique_ptr<Tile> buildCurrentTile() = 0;

private:
    TileIterBase(const TileIterBase&) = delete;
    TileIterBase& operator=(TileIterBase&&) = delete;
    // Limit all copies as much as possible
    TileIterBase(TileIterBase&) = delete;
    TileIterBase& operator=(TileIterBase&) = delete;
    TileIterBase& operator=(const TileIterBase&) = delete;
};

}

#endif

