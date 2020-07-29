#ifndef TILEITERBASE_H__
#define TILEITERBASE_H__

#include "Tile.h"

namespace orchestration {

class TileIterBase {
public:
    // TODO: implement tiling!!
    explicit TileIterBase(const unsigned int lev)
             : nodetype_{0},
               lev_{lev},
               currentIdx_{0},
               endIdx_{0}      {}

    TileIterBase(TileIterBase&&) = default;
    virtual ~TileIterBase(void) {}

    virtual bool isValid() const { return currentIdx_ < endIdx_; }
    virtual void operator++() { currentIdx_++; }
    virtual std::unique_ptr<Tile> buildCurrentTile() = 0;

protected:
    int currentIdx_;
    int endIdx_;
    int nodetype_;
    unsigned int lev_;

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

