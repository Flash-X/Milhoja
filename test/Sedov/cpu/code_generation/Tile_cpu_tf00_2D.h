#ifndef CGKIT_GENERATED_TILE_CPU_TF00_2D_H__
#define CGKIT_GENERATED_TILE_CPU_TF00_2D_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf00_2D : public milhoja::TileWrapper {
    Tile_cpu_tf00_2D(const milhoja::Real dt);
    ~Tile_cpu_tf00_2D(void);

    Tile_cpu_tf00_2D(Tile_cpu_tf00_2D&)                  = delete;
    Tile_cpu_tf00_2D(const Tile_cpu_tf00_2D&)            = delete;
    Tile_cpu_tf00_2D(Tile_cpu_tf00_2D&&)                 = delete;
    Tile_cpu_tf00_2D& operator=(Tile_cpu_tf00_2D&)       = delete;
    Tile_cpu_tf00_2D& operator=(const Tile_cpu_tf00_2D&) = delete;
    Tile_cpu_tf00_2D& operator=(Tile_cpu_tf00_2D&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const override;

    milhoja::Real  dt_;

    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  HYDRO_OP1_AUXC_SIZE_ =
                      18
                    * 18
                    * 1;

    static void* hydro_op1_auxc_;
};

#endif
