#ifndef MILHOJA_TILE_H__
#define MILHOJA_TILE_H__

#include "Milhoja_DataItem.h"
#include "Milhoja_FArray4D.h"
#include "Milhoja_IntVect.h"

namespace milhoja {

/**
 * \brief Provides access to pointers to physical data.
 *
 * When iterating over the domain, a Tile Iterator returns
 * Tiles which store indices to the correct location in the
 * physical data arrays. Tile inherits from DataItem. Tile is
 * an abstract class, each AMR package must implement its own
 * version of most of the member functions.
 *
 * \todo Create readonly versions of data getters?
 */
class Tile : public DataItem {
public:
    Tile(void);
    virtual ~Tile(void);

    Tile(Tile&)                  = delete;
    Tile(const Tile&)            = delete;
    Tile(Tile&&)                 = delete;
    Tile& operator=(Tile&)       = delete;
    Tile& operator=(const Tile&) = delete;
    Tile& operator=(Tile&&)      = delete;

    // Union of tile index information across all Grid backends.  Each backend
    // should be able to construct the Tile's unique index from a subset of
    // these.
    virtual unsigned int level(void) const = 0;
    virtual int          gridIndex(void) const = 0;
    virtual int          tileIndex(void) const = 0;

    virtual unsigned int        nCcVariables(void) const = 0;
    virtual unsigned int        nFluxVariables(void) const = 0;
    virtual RealVect            deltas(void) const;
    virtual IntVect             lo(void) const = 0;
    virtual IntVect             hi(void) const = 0;
    virtual IntVect             loGC(void) const = 0;
    virtual IntVect             hiGC(void) const = 0;
    virtual FArray4D            data(void) = 0;
    virtual Real*               dataPtr(void) = 0;
    virtual FArray4D            fluxData(const unsigned int dir) = 0;
    virtual std::vector<Real*>  fluxDataPtrs(void) = 0;

    virtual RealVect     getCenterCoords(void) const;
};

}

#endif

