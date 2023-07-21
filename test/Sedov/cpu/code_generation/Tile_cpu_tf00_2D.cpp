#include "Tile_cpu_tf00_2D.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

//----- STATIC SCRATCH MEMORY MANAGEMENT
void*  Tile_cpu_tf00_2D::auxC_scratch_ = nullptr;

/**
 *
 */
void   Tile_cpu_tf00_2D::acquireScratch(void) {
    if (auxC_scratch_) {
        throw std::logic_error("[Tile_cpu_tf00_2D::acquireScratch] "
                               "Scratch already allocated");
        
    }

    const unsigned int   nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();
    const std::size_t    nBytes =   nThreads
                                  * Tile_cpu_tf00_2D::AUXC_SIZE_
                                  * sizeof(milhoja::Real);
    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &auxC_scratch_);

#ifdef DEBUG_RUNTIME
    std::string   msg =   "[Tile_cpu_tf00_2D::acquireScratch] Acquired "
                        + std::to_string(nThreads)
                        + " auxC scratch blocks";
    milhoja::Logger::instance().log(msg);
#endif
}

/**
 *
 */
void   Tile_cpu_tf00_2D::releaseScratch(void) {
    if (!auxC_scratch_) {
        throw std::logic_error("[Tile_cpu_tf00_2D::releaseScratch] "
                               "auxC scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&auxC_scratch_);
    auxC_scratch_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string    msg = "[Tile_cpu_tf00_2D::releaseScratch] Released auxC scratch";
    milhoja::Logger::instance().log(msg);
#endif
}

/**
 * Instantiate a prototype of the wrapper.  The object does not wrap a tile and
 * is, therefore, typically only useful for cloning objects that do wrap the
 * actual tile given to clone().
 */
Tile_cpu_tf00_2D::Tile_cpu_tf00_2D(const milhoja::Real dt)
    : milhoja::TileWrapper{},
      dt_{dt}
{
}

/**
 *
 */
Tile_cpu_tf00_2D::~Tile_cpu_tf00_2D(void) {
#ifdef DEBUG_RUNTIME
    std::string msg = "[~Tile_cpu_tf00_2D] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

/**
 *
 */
std::unique_ptr<milhoja::TileWrapper>   Tile_cpu_tf00_2D::clone(
        std::unique_ptr<milhoja::Tile>&& tileToWrap) const {
    // Create new wrapper & set private data to prototype's values
    Tile_cpu_tf00_2D*   ptr = new Tile_cpu_tf00_2D{dt_};

    // New wrapper takes ownership of the tile to wrap
    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf00_2D::clone] "
                               "Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf00_2D::clone] "
                               "Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}

