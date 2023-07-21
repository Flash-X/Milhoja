#include "Milhoja_TileWrapper.h"

#ifdef DEBUG_RUNTIME
#include "Milhoja_Logger.h"
#endif

namespace milhoja {

/**
 * Instantiate a prototype of the wrapper for use with task functions that have
 * no external variables and, therefore, don't need a custom TileWrapper derived
 * class.  The object does not wrap a tile and is, therefore, typically only
 * useful for cloning objects that do wrap the actual tile given to clone().
 */
TileWrapper::TileWrapper(void)
    : tile_{}
{
}

TileWrapper::~TileWrapper(void) {
#ifdef DEBUG_RUNTIME
    std::string msg = "[TileWrapper] Destroying wrapper object";
    Logger::instance().log(msg);
#endif
}

std::unique_ptr<TileWrapper>   TileWrapper::clone(
        std::unique_ptr<Tile>&& tileToWrap) const {
    // New wrapper takes ownership of the tile to wrap
    TileWrapper*   ptr = new TileWrapper{};
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[TileWrapper::clone] "
                               "Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}

};

