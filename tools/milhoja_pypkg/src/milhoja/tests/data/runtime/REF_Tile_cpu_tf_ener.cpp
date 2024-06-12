#include "Tile_cpu_tf_ener.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_ener::scratch_base_op1_scratch3D_ = nullptr;

void Tile_cpu_tf_ener::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (scratch_base_op1_scratch3D_) {
        throw std::logic_error("[Tile_cpu_tf_ener::acquireScratch] scratch_base_op1_scratch3D_ scratch already allocated");
    }

    const std::size_t nBytes_scratch_base_op1_scratch3D = nThreads
                    * Tile_cpu_tf_ener::SCRATCH_BASE_OP1_SCRATCH3D_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes_scratch_base_op1_scratch3D, &scratch_base_op1_scratch3D_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_ener::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " scratch_base_op1_scratch3D_ scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf_ener::releaseScratch(void) {
    if (!scratch_base_op1_scratch3D_) {
        throw std::logic_error("[Tile_cpu_tf_ener::releaseScratch] scratch_base_op1_scratch3D_ scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&scratch_base_op1_scratch3D_);
    scratch_base_op1_scratch3D_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_ener::releaseScratch] Released scratch_base_op1_scratch3D_ scratch"
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_ener::Tile_cpu_tf_ener(void)
    : milhoja::TileWrapper{}
{
#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_ener] Creating wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_ener::~Tile_cpu_tf_ener(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_ener] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_ener::clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_ener* ptr = new Tile_cpu_tf_ener{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_ener::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_ener::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
