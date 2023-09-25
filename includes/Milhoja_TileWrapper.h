/**
 * \file Milhoja_TileWrapper.h
 *
 * This class was designed using the Prototype design pattern (Citation XXX) so
 * that given a prototype TileWrapper object of an unknown concrete type, the
 * runtime can clone this and therefore create new TileWrapper objects of the
 * appropriate concrete type.  Note that the runtime is therefore decoupled from
 * the fine details of each TileWrapper --- it is effectively a conduit.  Given a
 * prototype, the runtime blindly creates TileWrapper objects that know how to
 * construct themselves and passes the objects to a task function that knows how
 * to use them.  This design is intentionally similar to the prototype design of
 * the DataPacket interface.
 */

#ifndef MILHOJA_TILE_WRAPPER_H__
#define MILHOJA_TILE_WRAPPER_H__

#include "Milhoja_Tile.h"

namespace milhoja {

/**
 * \brief 
 */
struct TileWrapper : public DataItem {
public:
    TileWrapper(void);
    virtual ~TileWrapper(void);

    TileWrapper(TileWrapper&)                  = delete;
    TileWrapper(const TileWrapper&)            = delete;
    TileWrapper(TileWrapper&&)                 = delete;
    TileWrapper& operator=(TileWrapper&)       = delete;
    TileWrapper& operator=(const TileWrapper&) = delete;
    TileWrapper& operator=(TileWrapper&&)      = delete;

    /**
     * The functionality of this function is similar to the combined
     * functionality of the clone() and addTile() functions of the DataPacket
     * class.
     *
     * Obtain a pointer to a new TileWrapper of the same concrete type as the
     * calling object and that wraps the given tile object.  The main workhorse
     * of the Prototype design pattern.  The new wrapper also contains *copies*
     * of the variables used to instantiate the prototype.  Note that the
     * wrapper assumes ownership of the Tile and the calling code's pointer is
     * nullified.
     *
     * @param tileToWrap - the tile resource to wrap.  As part of the memory
     * resource management scheme, this must be a shared_ptr since the same tile
     * object could be packaged up into a different DataItem (e.g., a data
     * packet for the GPU) that flows through a different pipeline concurrently.
     *
     * @return The pointer.  For the main use cases in the runtime, this should
     * be cast to a shared_ptr.  We return a unique_ptr based on the discussion
     * in Item 19 (Pp 113) of Effective Modern C++.
     *
     * @todo Add in citation.
     */
    virtual std::unique_ptr<TileWrapper>  clone(std::shared_ptr<Tile>&& tileToWrap) const;

    std::shared_ptr<Tile>  tile_;
};

}

#endif

