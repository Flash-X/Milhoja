#include "FArray4D.h"

#include <stdexcept>

namespace orchestration {

//----- static member function definitions
//! Factory function to build an FArray4D that owns data.
FArray4D   FArray4D::buildScratchArray4D(const IntTriple& lo,
                                         const IntTriple& hi,
                                         const unsigned int ncomp) {
    Real*  scratch = new Real[   (hi[Axis::I] - lo[Axis::I] + 1)
                               * (hi[Axis::J] - lo[Axis::J] + 1)
                               * (hi[Axis::K] - lo[Axis::K] + 1)
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
                   const IntTriple& lo, const IntTriple& hi,
                   const unsigned int ncomp)
    : owner_{false},
      data_{data},
      i0_{lo[Axis::I]},
      j0_{lo[Axis::J]},
      k0_{lo[Axis::K]},
      jstride_(         (hi[Axis::I] - i0_ + 1)),
      kstride_(jstride_*(hi[Axis::J] - j0_ + 1)),
      nstride_(kstride_*(hi[Axis::K] - k0_ + 1)),
      ncomp_{ncomp}
{
#ifndef GRID_ERRCHECK_OFF
    // TODO Add tests for these?
    if (!data_) {
        throw std::invalid_argument("[FArray4D::FArray4D] null data pointer given");
    } else if (ncomp_ == 0) {
        throw std::invalid_argument("[FArray4D::FArray4D] empty array specified");
    } else if (   (lo[Axis::I] > hi[Axis::I])
               || (lo[Axis::J] > hi[Axis::J])
               || (lo[Axis::K] > hi[Axis::K]) ) {
        std::string   msg =   "[FArray4D::FArray4D] lo ("
                            + std::to_string(lo[Axis::I]) + ", "
                            + std::to_string(lo[Axis::J]) + ", "
                            + std::to_string(lo[Axis::K]) + ") ";
                            + " not compatible with hi ("
                            + std::to_string(hi[Axis::I]) + ", "
                            + std::to_string(hi[Axis::J]) + ", "
                            + std::to_string(hi[Axis::K]) + ") ";
        throw std::invalid_argument(msg);
    }
#endif
}

//! Destructor that deletes data if it owns it.
FArray4D::~FArray4D(void) {
    if (owner_) {
        delete [] data_;
    }
    data_ = nullptr;
}

}

