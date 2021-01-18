#include "FArray3D.h"

#include <stdexcept>

namespace orchestration {

//----- static member function definitions
/** Factory function to build an FArray3D that owns data
 *  The objects obtained by calling this function must never be copied to a
 *  DataPacket as the destruction of the objects would lead to releasing
 *  memory to be used later by the copies.
 */
FArray3D   FArray3D::buildScratchArray(const IntVect& lo, const IntVect& hi) {
    Real*  scratch = new Real[   (hi.I() - lo.I() + 1)
                               * (hi.J() - lo.J() + 1)
                               * (hi.K() - lo.K() + 1)];
    FArray3D   scratchArray = FArray3D{scratch, lo, hi};
    scratchArray.owner_ = true;
    return scratchArray;
}

//----- member function definitions
/** \brief Construct an FArray3D to wrap an input pointer.
  *
  * The three dimensions are defined by lo and hi.  The user is responsible for
  * ensuring size of data matches lo and hi.
  */
FArray3D::FArray3D(Real* data, const IntVect& lo, const IntVect& hi)
    : owner_{false},
      data_{data},
      i0_{lo.I()},
      j0_{lo.J()},
      k0_{lo.K()},
      jstride_(         (hi.I() - i0_ + 1)),
      kstride_(jstride_*(hi.J() - j0_ + 1))
{
#ifndef GRID_ERRCHECK_OFF
    // TODO Add tests for these?
    if (!data_) {
        throw std::invalid_argument("[FArray3D::FArray3D] null data pointer given");
    } else if (   (lo.I() > hi.I())
               || (lo.J() > hi.J())
               || (lo.K() > hi.K()) ) {
        std::string   msg =   "[FArray3D::FArray3D] lo ("
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
FArray3D::~FArray3D(void) {
    if (owner_) {
        delete [] data_;
    }
    data_ = nullptr;
}

}

