#ifndef TILE_CPU_TF00_3D_H__
#define TILE_CPU_TF00_3D_H__

#include <Milhoja_TileWrapper.h>

struct Tile_cpu_tf00_3D : public milhoja::TileWrapper {
    Tile_cpu_tf00_3D(const milhoja::Real dt);
    ~Tile_cpu_tf00_3D(void);

    Tile_cpu_tf00_3D(Tile_cpu_tf00_3D&)                  = delete;
    Tile_cpu_tf00_3D(const Tile_cpu_tf00_3D&)            = delete;
    Tile_cpu_tf00_3D(Tile_cpu_tf00_3D&&)                 = delete;
    Tile_cpu_tf00_3D& operator=(Tile_cpu_tf00_3D&)       = delete;
    Tile_cpu_tf00_3D& operator=(const Tile_cpu_tf00_3D&) = delete;
    Tile_cpu_tf00_3D& operator=(Tile_cpu_tf00_3D&&)      = delete;

    // TODO: If all CPU task functions end up using a dedicated 
    //       Tile class, then this should be an override of a method higher up
    //       in the class hierarchy.  Likely, it should be declared in DataItem.
    std::unique_ptr<milhoja::TileWrapper> clone(std::unique_ptr<milhoja::Tile>&& tileToWrap) const override;

    // Thread-private variables for direct access by task function
    milhoja::Real     dt_;

    // Thread-team private scratch memory for direct access by task function
    // The scratch is accessed by thread ID so that each block in the pool is
    // thread-private as well.
    //
    // For now, we will acquire/release scratch with each invocation.  But we
    // could come up with a means for users to specify if the scratch should be
    // allocated at the start of the simulation or at each cycle.
    static void   acquireScratch(void);
    static void   releaseScratch(void);

    static milhoja::Real*  auxC_scratch_;
};

#endif
