#ifndef TILEITER_H__
#define TILEITER_H__

#include "TileIterBase.h"

namespace orchestration {

/**
  * Wrapper class for std::unique_ptr<TileIterBase>.
  */
class TileIter {
public:
    explicit TileIter( std::unique_ptr<TileIterBase> tiIn) : tiPtr_{std::move(tiIn)} {}
    TileIter(TileIter&&) = default;

    bool isValid() const { return tiPtr_->isValid(); }
    void operator++() { ++(*tiPtr_); }
    std::unique_ptr<Tile> buildCurrentTile() { return tiPtr_->buildCurrentTile(); }

private:
    std::unique_ptr<TileIterBase> tiPtr_;

    TileIter(const TileIter&) = delete;
    TileIter& operator=(TileIter&&) = delete;
    // Limit all copies as much as possible
    TileIter(TileIter&) = delete;
    TileIter& operator=(TileIter&) = delete;
    TileIter& operator=(const TileIter&) = delete;
};

}

#endif

