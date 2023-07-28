#include "Tile_cpu_tf00_3D.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

//----- STATIC SCRATCH MEMORY MANAGEMENT
void*  Tile_cpu_tf00_3D::hydro_op1_auxc_ = nullptr;

/**
 *
 */
void   Tile_cpu_tf00_3D::acquireScratch(void) {
    if (hydro_op1_auxc_) {
        throw std::logic_error("[Tile_cpu_tf00_3D::acquireScratch] "
                               "Scratch already allocated");
        
    }

    const unsigned int   nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();
    const std::size_t    nBytes =   nThreads
                                  * Tile_cpu_tf00_3D::hydro_op1_auxc_SIZE_
                                  * sizeof(milhoja::Real);
    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &hydro_op1_auxc_);

#ifdef DEBUG_RUNTIME
    std::string   msg =   "[Tile_cpu_tf00_3D::acquireScratch] Acquired "
                        + std::to_string(nThreads)
                        + " auxC scratch blocks";
    milhoja::Logger::instance().log(msg);
#endif
}

/**
 *
 */
void   Tile_cpu_tf00_3D::releaseScratch(void) {
    if (!hydro_op1_auxc_) {
        throw std::logic_error("[Tile_cpu_tf00_3D::releaseScratch] "
                               "auxC scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&hydro_op1_auxc_);
    hydro_op1_auxc_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string    msg = "[Tile_cpu_tf00_3D::releaseScratch] Released auxC scratch";
    milhoja::Logger::instance().log(msg);
#endif
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

