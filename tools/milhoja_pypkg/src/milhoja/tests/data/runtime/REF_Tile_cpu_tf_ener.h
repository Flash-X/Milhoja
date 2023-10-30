#ifndef MILHOJA_GENERATED_TILE_CPU_TF_ENER_H__
#define MILHOJA_GENERATED_TILE_CPU_TF_ENER_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_ener : public milhoja::TileWrapper {
    Tile_cpu_tf_ener(void);
    ~Tile_cpu_tf_ener(void);

    Tile_cpu_tf_ener(Tile_cpu_tf_ener&)                  = delete;
    Tile_cpu_tf_ener(const Tile_cpu_tf_ener&)            = delete;
    Tile_cpu_tf_ener(Tile_cpu_tf_ener&&)                 = delete;
    Tile_cpu_tf_ener& operator=(Tile_cpu_tf_ener&)       = delete;
    Tile_cpu_tf_ener& operator=(const Tile_cpu_tf_ener&) = delete;
    Tile_cpu_tf_ener& operator=(Tile_cpu_tf_ener&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  BASE_OP1_SCRATCH3D_SIZE_ =
                      8
                    * 16
                    * 1;

    static void* base_op1_scratch3D_;
};

#endif
