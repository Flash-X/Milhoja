#include "FArray4D.h"

#include <stdexcept>

namespace orchestration {

//----- static member function definitions
//! Factory function to build an FArray4D that owns data.
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

//! Destructor that deletes data if it owns it.
FArray4D::~FArray4D(void) {
    if (owner_) {
        delete [] data_;
    }
    data_ = nullptr;
}

}

