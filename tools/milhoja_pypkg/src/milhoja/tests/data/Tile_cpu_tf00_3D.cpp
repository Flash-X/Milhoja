#include "Tile_delete_me.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif

void*  Tile_delete_me::hydro_op1_auxc_ = nullptr;

void Tile_delete_me::acquireScratch(void) {
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

    if (hydro_op1_auxc_) {
        throw std::logic_error("[Tile_delete_me::acquireScratch] hydro_op1_auxc scratch already allocated");
    }

    const std::size_t nBytes = nThreads
                    * Tile_delete_me::HYDRO_OP1_AUXC_SIZE_
                    * sizeof(milhoja::Real);

    milhoja::RuntimeBackend::instance().requestCpuMemory(nBytes, &hydro_op1_auxc_);

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_delete_me::acquireScratch] Acquired"
                    + std::to_string(nThreads)
                    + " hydro_op1_auxc scratch blocks"
    milhoja::Logger::instance().log(msg);
#endif
}

void Tile_delete_me::releaseScratch(void) {
    if (!hydro_op1_auxc_) {
        throw std::logic_error("[Tile_delete_me::releaseScratch] hydro_op1_auxc scratch not allocated");
    }

    milhoja::RuntimeBackend::instance().releaseCpuMemory(&hydro_op1_auxc_);
    hydro_op1_auxc_ = nullptr;

#ifdef DEBUG_RUNTIME
    std::string   msg = "[Tile_delete_me::releaseScratch] Released hydro_op1_auxc scratch"
    milhoja::Logger::instance().log(msg);
#endif
}

Tile_delete_me::Tile_delete_me(const milhoja::Real dt)
    : milhoja::TileWrapper{},
      dt_{dt}
{
}

Tile_delete_me::~Tile_delete_me(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_delete_me] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_delete_me::clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_delete_me* ptr = new Tile_delete_me{dt_};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_delete_me::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_delete_me::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
