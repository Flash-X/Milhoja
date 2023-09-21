#include "Tile_cpu_tf_ic.h"

#include <Milhoja_Runtime.h>
#include <Milhoja_RuntimeBackend.h>
#ifdef DEBUG_RUNTIME
#include <Milhoja_Logger.h>
#endif


void Tile_cpu_tf_ic::acquireScratch(void) {
    // TODO: If we can't get rid of these member functions, at least have them
    // be no-ops.
    const unsigned int  nThreads = milhoja::Runtime::instance().nMaxThreadsPerTeam();

}

void Tile_cpu_tf_ic::releaseScratch(void) {
}

Tile_cpu_tf_ic::Tile_cpu_tf_ic(void)
    : milhoja::TileWrapper{}
{
// TODO: Log something here?
}

Tile_cpu_tf_ic::~Tile_cpu_tf_ic(void) {
#ifdef DEBUG_RUNTIME
    std::string   msg = "[~Tile_cpu_tf_ic] Destroying wrapper object";
    milhoja::Logger::instance().log(msg);
#endif
}

std::unique_ptr<milhoja::TileWrapper> Tile_cpu_tf_ic::clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const {
    Tile_cpu_tf_ic* ptr = new Tile_cpu_tf_ic{};

    if (ptr->tile_) {
        throw std::logic_error("[Tile_cpu_tf_ic::clone] Internal tile_ member not null");
    }
    ptr->tile_ = std::move(tileToWrap);
    if (!(ptr->tile_) || tileToWrap) {
        throw std::logic_error("[Tile_cpu_tf_ic::clone] Wrapper did not take ownership of tile");
    }

    return std::unique_ptr<milhoja::TileWrapper>{ptr};
}
