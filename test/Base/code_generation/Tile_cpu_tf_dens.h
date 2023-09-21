#ifndef CGKIT_GENERATED_TILE_CPU_TF_DENS_H__
#define CGKIT_GENERATED_TILE_CPU_TF_DENS_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_dens : public milhoja::TileWrapper {
    Tile_cpu_tf_dens(void);
    ~Tile_cpu_tf_dens(void);

    Tile_cpu_tf_dens(Tile_cpu_tf_dens&)                  = delete;
    Tile_cpu_tf_dens(const Tile_cpu_tf_dens&)            = delete;
    Tile_cpu_tf_dens(Tile_cpu_tf_dens&&)                 = delete;
    Tile_cpu_tf_dens& operator=(Tile_cpu_tf_dens&)       = delete;
    Tile_cpu_tf_dens& operator=(const Tile_cpu_tf_dens&) = delete;
    Tile_cpu_tf_dens& operator=(Tile_cpu_tf_dens&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  BASE_OP1_SCRATCH_SIZE_ =
                      8
                    * 16
                    * 1
                    * 1;

    static void* base_op1_scratch_;
};

#endif
