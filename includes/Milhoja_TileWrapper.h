#ifndef MILHOJA_TILE_WRAPPER_H__
#define MILHOJA_TILE_WRAPPER_H__

#include "Milhoja_Tile.h"

namespace milhoja {

/**
 * \brief 
 */
struct TileWrapper : public DataItem {
public:
    TileWrapper(std::unique_ptr<Tile>&& tileToWrap);
    virtual ~TileWrapper(void);

    TileWrapper(TileWrapper&)                  = delete;
    TileWrapper(const TileWrapper&)            = delete;
    TileWrapper(TileWrapper&&)                 = delete;
    TileWrapper& operator=(TileWrapper&)       = delete;
    TileWrapper& operator=(const TileWrapper&) = delete;
    TileWrapper& operator=(TileWrapper&&)      = delete;

    virtual std::unique_ptr<TileWrapper>  clone(std::unique_ptr<Tile>&& tileToWrap) const;

    std::unique_ptr<Tile>  tile_;
};

}

#endif

