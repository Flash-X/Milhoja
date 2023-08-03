#include "Tile_cpu_tf00_3D.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf00_3D::hydro_op1_auxc_ = nullptr;

void Tile_cpu_tf00_3D::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (hydro_op1_auxc_) {
        throw std::logic_error("[Tile_cpu_tf00_3D::acquireScratch] hydro_op1_auxc scratch already allocated");
    }

    const std::size_t nBytes = nThreads
                    * Tile_cpu_tf00_3D::HYDRO_OP1_AUXC_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &hydro_op1_auxc_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf00_3D::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " hydro_op1_auxc scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf00_3D::releaseScratch(void) {
    if (!hydro_op1_auxc_) {
        throw std::logic_error("[Tile_cpu_tf00_3D::releaseScratch] hydro_op1_auxc scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&hydro_op1_auxc_);
    hydro_op1_auxc_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf00_3D::releaseScratch] Released hydro_op1_auxc scratch"
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf00_3D::Tile_cpu_tf00_3D(const milhoja::Real dt)
    : milhoja::TileWrapper{},
      dt_{dt}
{
}

Tile_cpu_tf00_3D::~Tile_cpu_tf00_3D(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf00_3D] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf00_3D::clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf00_3D* ptr = new Tile_cpu_tf00_3D{dt_};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf00_3D::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf00_3D::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
