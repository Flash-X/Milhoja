#ifndef CGKIT_GENERATED_TILE_CPU_TF_IQ_2D_H__
#define CGKIT_GENERATED_TILE_CPU_TF_IQ_2D_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf_IQ_2D : public milhoja::TileWrapper {
    Tile_cpu_tf_IQ_2D(void);
    ~Tile_cpu_tf_IQ_2D(void);

    Tile_cpu_tf_IQ_2D(Tile_cpu_tf_IQ_2D&)                  = delete;
    Tile_cpu_tf_IQ_2D(const Tile_cpu_tf_IQ_2D&)            = delete;
    Tile_cpu_tf_IQ_2D(Tile_cpu_tf_IQ_2D&&)                 = delete;
    Tile_cpu_tf_IQ_2D& operator=(Tile_cpu_tf_IQ_2D&)       = delete;
    Tile_cpu_tf_IQ_2D& operator=(const Tile_cpu_tf_IQ_2D&) = delete;
    Tile_cpu_tf_IQ_2D& operator=(Tile_cpu_tf_IQ_2D&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::shared_ptr<milhoja::Tile>&& tileToWrap) const override;


    static void acquireScratch(void);
    static void releaseScratch(void);

    // TODO: Presently allocating 3D blocks because the code generator isn't
    // written to manage different dimensions yet.
    constexpr static std::size_t  _MH_INTERNAL_VOLUMES_SIZE_ =
                      18
                    * 18
                    * 18;

    static void* _mh_internal_volumes_;
};

#endif
