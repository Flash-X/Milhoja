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

    virtual bool isValid() const { return currentIdx_ < endIdx_; }
    virtual void operator++() { currentIdx_++; }
    virtual std::unique_ptr<Tile> buildCurrentTile() = 0;

protected:
    int currentIdx_;
    int endIdx_;
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

