#include "Milhoja_TileWrapper.h"

#include "Milhoja_Logger.h"

namespace milhoja {

TileWrapper::TileWrapper(std::unique_ptr<Tile>&& tileToWrap)
    : tile_{ std::move(tileToWrap) }
{
#ifdef DEBUG_RUNTIME
    if (tile_) {
        std::string msg = "[TileWrapper] Wrapping pointer";
        Logger::instance().log(msg);
    } else {
        std::string msg = "[TileWrapper] Wrapping null pointer";
        Logger::instance().log(msg);
    }
#endif
}

TileWrapper::~TileWrapper(void) {
#ifdef DEBUG_RUNTIME
    std::string msg = "[TileWrapper] Destroying wrapper object";
    Logger::instance().log(msg);
#endif
}

std::unique_ptr<TileWrapper>   TileWrapper::clone(
        std::unique_ptr<Tile>&& tileToWrap) const {
    if (!tileToWrap) {
        throw std::logic_error("[TileWrapper::clone] "
                               "Tile to wrap is null");
    }

    // New wrapper takes ownership of the tile to wrap
    TileWrapper*   ptr = new TileWrapper{ std::move(tileToWrap) };
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[TileWrapper::clone] "
                               "Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}

};

