#include "Milhoja_FArray4D.h"

#include <stdexcept>

namespace milhoja {

//----- static member function definitions
/** Factory function to build an FArray4D that owns data
 *  The objects obtained by calling this function must never be copied to a
 *  DataPacket as the destruction of the objects would lead to releasing
 *  memory to be used later by the copies.
 */
FArray4D   FArray4D::buildScratchArray4D(const IntVect& lo,
                                         const IntVect& hi,
                                         const unsigned int ncomp) {
    Real*  scratch = new Real[   (hi.I() - lo.I() + 1)
                               * (hi.J() - lo.J() + 1)
                               * (hi.K() - lo.K() + 1)
                               *  ncomp];
    FArray4D   scratchArray = FArray4D{scratch, lo, hi, ncomp};
    scratchArray.owner_ = true;
    return scratchArray;
}

//----- member function definitions
/** \brief Construct an FArray4D to wrap an input pointer.
  *
  * The first three dimensions are defined by lo and hi, and the
  * fourth dimension is defined by ncomp. User is responsible for ensuring
  * size of data matches lo, hi, and ncomp.
  */
FArray4D::FArray4D(Real* data, 
                   const IntVect& lo, const IntVect& hi,
                   const unsigned int ncomp)
    : owner_{false},
      data_{data},
      i0_{lo.I()},
      j0_{lo.J()},
      k0_{lo.K()},
      jstride_(         (hi.I() - i0_ + 1)),
      kstride_(jstride_*(hi.J() - j0_ + 1)),
      nstride_(kstride_*(hi.K() - k0_ + 1)),
      ncomp_{ncomp}
{
#ifndef GRID_ERRCHECK_OFF
    // TODO Add tests for these?
    if (!data_) {
        throw std::invalid_argument("[FArray4D::FArray4D] null data pointer given");
    } else if (ncomp_ == 0) {
        throw std::invalid_argument("[FArray4D::FArray4D] empty array specified");
    } else if (   (lo.I() > hi.I())
               || (lo.J() > hi.J())
               || (lo.K() > hi.K()) ) {
        std::string   msg =   "[FArray4D::FArray4D] lo ("
                            + std::to_string(lo.I()) + ", "
                            + std::to_string(lo.J()) + ", "
                            + std::to_string(lo.K()) + ") ";
                            + " not compatible with hi ("
                            + std::to_string(hi.I()) + ", "
                            + std::to_string(hi.J()) + ", "
                            + std::to_string(hi.K()) + ") ";
        throw std::invalid_argument(msg);
    }
#endif
}

/** Destructor that deletes data if it owns it.
 * IMPORTANT: Copies of objects of this type will be copied into data packets,
 * which could persist for longer than the original object.  Therefore, the 
 * destructor of this class must never perform any actions that could prevent
 * the data packet copies from functioning correctly (e.g. releasing memory).
 */
FArray4D::~FArray4D(void) {
    if (owner_) {
        delete [] data_;
    }
    data_ = nullptr;
}

/**
 * While the elements of scratch arrays must be assigned a given index set at
 * creation, conceptually they are not tied to any specific piece of data nor
 * any particular region of an index space.  FArray objects created around a
 * given array, however, are conceptually tied to one particular piece of data
 * and, therefore, to the index set associated with the data.
 * 
 * To address the use case of a single scratch array that needs to be reused to
 * store data associated with different index sets, this function allows calling
 * code to reindex the scratch array, which avoids the wasteful scenario of
 * deallocating/reallocating scratch arrays.  Base on the conceptual difference
 * between the different types of FArray objects, it is a logical error to
 * reindex a non-scratch array.
 * 
 * @param lo - the new index of the lower corner of the scratch array.
 */
void  FArray4D::reindex(const IntVect& lo) {
    if (owner_) {
        i0_ = lo.I();
        j0_ = lo.J();
        k0_ = lo.K();
    } else {
        throw std::logic_error("[FArray4D::reindex] Reindexing only allowed for scratch arrays");
    }
}

}

