#ifndef TILEITER_H__
#define TILEITER_H__

#include "Tile.h"

namespace orchestration {

class TileIter {
public:
    explicit TileIter(const unsigned int lev, const bool use_tiling=false)
             : nodetype_{0},
               lev_{lev},
               currentIdx_{0},
               endIdx_{0}      {}

    TileIter(TileIter&&) = default;
    virtual ~TileIter(void) {}

    virtual bool isValid() const { return currentIdx_ < maxIdx_; }
    virtual void operator++() { currentIdx_++; }
    virtual Tile currentTile() = 0;

protected:
    int currentIdx_;
    int maxIdx_;
    int nodetype_;
    unsigned int lev_;

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

