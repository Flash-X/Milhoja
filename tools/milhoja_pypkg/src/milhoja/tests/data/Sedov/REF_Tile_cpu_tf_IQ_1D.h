#ifndef MILHOJA_GENERATED_TILE_CPU_TF_IQ_H__
#define MILHOJA_GENERATED_TILE_CPU_TF_IQ_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_IQ : public milhoja::TileWrapper {
    Tile_cpu_tf_IQ(void);
    ~Tile_cpu_tf_IQ(void);

    Tile_cpu_tf_IQ(Tile_cpu_tf_IQ&)                  = delete;
    Tile_cpu_tf_IQ(const Tile_cpu_tf_IQ&)            = delete;
    Tile_cpu_tf_IQ(Tile_cpu_tf_IQ&&)                 = delete;
    Tile_cpu_tf_IQ& operator=(Tile_cpu_tf_IQ&)       = delete;
    Tile_cpu_tf_IQ& operator=(const Tile_cpu_tf_IQ&) = delete;
    Tile_cpu_tf_IQ& operator=(Tile_cpu_tf_IQ&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  MH_INTERNAL_CELLVOLUMES_SIZE_ =
                      16
                    * 1
                    * 1;

    static void* MH_INTERNAL_cellVolumes_;
};

#endif
