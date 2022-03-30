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
 */
class Tile : public DataItem {
public:
    Tile(void);
    virtual ~Tile(void);

    Tile(Tile&&) = delete;
    Tile& operator=(Tile&&) = delete;
    Tile(Tile&) = delete;
    Tile(const Tile&) = delete;
    Tile& operator=(Tile&) = delete;
    Tile& operator=(const Tile&) = delete;

    // Pure virtual functions
    virtual bool         isNull(void) const = 0;
    virtual int          gridIndex(void) const = 0;
    virtual unsigned int level(void) const = 0;
    virtual unsigned int nVariables(void) const = 0;
    virtual IntVect      lo(void) const = 0;
    virtual IntVect      hi(void) const = 0;
    virtual IntVect      loGC(void) const = 0;
    virtual IntVect      hiGC(void) const = 0;
    virtual void         lo(int* i, int* j, int* k) const = 0;
    virtual void         hi(int* i, int* j, int* k) const = 0;
    virtual void         loGC(int* i, int* j, int* k) const = 0;
    virtual void         hiGC(int* i, int* j, int* k) const = 0;

    // TODO: Create readonly versions of these?
    virtual FArray4D     data(void) = 0;
    virtual Real*        dataPtr(void) = 0;

    // Virtual functions with a default implementation.
    virtual RealVect     deltas(void) const;
    virtual RealVect     getCenterCoords(void) const;
};

}

#endif

