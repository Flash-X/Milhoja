#ifndef CGKIT_GENERATED_TILE_DELETE_ME_H__
#define CGKIT_GENERATED_TILE_DELETE_ME_H__

#include <Milhoja_TileWrapper.h>

struct Tile_delete_me : public milhoja::TileWrapper {
    Tile_delete_me(const milhoja::Real dt);
    ~Tile_delete_me(void);

    Tile_delete_me(Tile_delete_me&)                  = delete;
    Tile_delete_me(const Tile_delete_me&)            = delete;
    Tile_delete_me(Tile_delete_me&&)                 = delete;
    Tile_delete_me& operator=(Tile_delete_me&)       = delete;
    Tile_delete_me& operator=(const Tile_delete_me&) = delete;
    Tile_delete_me& operator=(Tile_delete_me&&)      = delete;

    std::unique_ptr<milhoja::TileWrapper> clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const override;

    milhoja::Real  dt_;

    static void acquireScratch(void);
    static void releaseScratch(void);

    constexpr static std::size_t  HYDRO_OP1_AUXC_SIZE_ =
                      10
                    * 10
                    * 10;

    static void* hydro_op1_auxc_;
};

#endif
