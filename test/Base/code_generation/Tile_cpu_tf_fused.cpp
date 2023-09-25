#include "Tile_cpu_tf_fused.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_cpu_tf_fused::base_op1_scratch_ = nullptr;

void Tile_cpu_tf_fused::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (base_op1_scratch_) {
        throw std::logic_error("[Tile_cpu_tf_fused::acquireScratch] base_op1_scratch scratch already allocated");
    }

    const std::size_t nBytes = nThreads
                    * Tile_cpu_tf_fused::BASE_OP1_SCRATCH_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &base_op1_scratch_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_fused::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " base_op1_scratch scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_cpu_tf_fused::releaseScratch(void) {
    if (!base_op1_scratch_) {
        throw std::logic_error("[Tile_cpu_tf_fused::releaseScratch] base_op1_scratch scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&base_op1_scratch_);
    base_op1_scratch_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_cpu_tf_fused::releaseScratch] Released base_op1_scratch scratch"
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_cpu_tf_fused::Tile_cpu_tf_fused(void)
    : milhoja::TileWrapper{}
{
}

Tile_cpu_tf_fused::~Tile_cpu_tf_fused(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_fused] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_fused::clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_fused* ptr = new Tile_cpu_tf_fused{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_fused::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_fused::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
