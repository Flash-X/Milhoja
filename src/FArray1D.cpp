#include "FArray1D.h"

#include <stdexcept>
#include <string>

namespace orchestration {

//----- static member function definitions
//! Factory function to build an FArray1D that owns data.
FArray1D   FArray1D::buildScratchArray1D(const int lo,
                                         const int hi) {
    Real*  scratch = new Real[ (hi - lo + 1) ];
    FArray1D   scratchArray = FArray1D{scratch, lo, hi};
    scratchArray.owner_ = true;
    return scratchArray;
}

//----- member function definitions
/** \brief Construct an FArray1D to wrap an input pointer.
  *
  */
FArray1D::FArray1D(Real* data, 
                   const int lo, const int hi)
    : owner_{false},
      data_{data},
      i0_{lo}
{
#ifndef GRID_ERRCHECK_OFF
    // TODO Add tests for these?
    if (!data_) {
        throw std::invalid_argument("[FArray1D::FArray1D] null data pointer given");
    } else if (lo >= hi ) {
        std::string   msg =   "[FArray1D::FArray1D] lo ("
                            + std::to_string(lo) + ")"
                            + " not compatible with hi ("
                            + std::to_string(hi) + ") ";
        throw std::invalid_argument(msg);
    }
#endif
}

//! Destructor that deletes data if it owns it.
FArray1D::~FArray1D(void) {
    if (owner_) {
        delete [] data_;
    }
    data_ = nullptr;
}

}

