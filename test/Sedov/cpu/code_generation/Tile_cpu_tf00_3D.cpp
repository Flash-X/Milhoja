#include "Tile_cpu_tf00_3D.h"

#include <iostream>

//----- STATIC SCRATCH MEMORY MANAGEMENT
milhoja::Real*  Tile_cpu_tf00_3D::auxC_scratch_ = nullptr;

/**
 *
 */
void   Tile_cpu_tf00_3D::acquireScratch(void) {
    if (auxC_scratch_) {
        throw std::logic_error("[Tile_cpu_tf00_3D::acquireScratch] "
                               "Scratch already allocated");
        
    }

    // TODO: Acquire from runtime backend
    std::cout << "Allocated scratch for Tile_cpu_tf00_3D" << std::endl;
}

/**
 *
 */
void   Tile_cpu_tf00_3D::releaseScratch(void) {
//    if (!auxC_scratch_) {
//        throw std::logic_error("[Tile_cpu_tf00_3D::releaseScratch] "
//                               "Scratch not allocated");
//    }

    // TODO: Release with runtime backend
    std::cout << "Scratch for Tile_cpu_tf00_3D released" << std::endl;
}

/**
 * Instantiate a prototype of the wrapper.  The object does not wrap a tile and
 * is, therefore, typically only useful for cloning objects that do wrap the
 * actual tile given to clone().
 */
Tile_cpu_tf00_3D::Tile_cpu_tf00_3D(const milhoja::Real dt)
    : milhoja::TileWrapper{},
      dt_{dt}
{
}

/**
 *
 */
Tile_cpu_tf00_3D::~Tile_cpu_tf00_3D(void) {
#ifdef DEBUG_RUNTIME
    std::string msg = "[~Tile_cpu_tf00_3D] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

/**
 *
 */
std::unique_ptr<milhoja::TileWrapper>   Tile_cpu_tf00_3D::clone(
        std::unique_ptr<milhoja::Tile>&& tileToWrap) const {
    // Create new wrapper & set private data to prototype's values
    Tile_cpu_tf00_3D*   ptr = new Tile_cpu_tf00_3D{dt_};

    // New wrapper takes ownership of the tile to wrap
    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf00_3D::clone] "
                               "Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf00_3D::clone] "
                               "Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}

