#ifndef MILHOJA_GENERATED_TILE_CPU_TF_FUSED_H__
#define MILHOJA_GENERATED_TILE_CPU_TF_FUSED_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_fused : public milhoja::TileWrapper {
    Tile_cpu_tf_fused(void);
    ~Tile_cpu_tf_fused(void);

    Tile_cpu_tf_fused(Tile_cpu_tf_fused&)                  = delete;
    Tile_cpu_tf_fused(const Tile_cpu_tf_fused&)            = delete;
    Tile_cpu_tf_fused(Tile_cpu_tf_fused&&)                 = delete;
    Tile_cpu_tf_fused& operator=(Tile_cpu_tf_fused&)       = delete;
    Tile_cpu_tf_fused& operator=(const Tile_cpu_tf_fused&) = delete;
    Tile_cpu_tf_fused& operator=(Tile_cpu_tf_fused&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  SCRATCH_BASE_OP1_SCRATCH4D_SIZE_ =
                      8
                    * 16
                    * 1
                    * 2;

    static void* scratch_base_op1_scratch4D_;
};

#endif
