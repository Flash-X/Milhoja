#include "FArray2D.h"

#include <stdexcept>

namespace orchestration {

//----- member function definitions
/** \brief Construct an FArray2D to wrap an input pointer.
  *
  * The first two dimensions are defined by lo and hi.  User is responsible for
  * ensuring size of data matches lo, hi.
  */
FArray2D::FArray2D(Real* data, const IntVect& lo, const IntVect& hi)
    : data_{data},
      i0_{lo.I()},
      j0_{lo.J()},
      jstride_(hi.I() - i0_ + 1)
{
#ifndef GRID_ERRCHECK_OFF
    // TODO Add tests for these?
    if (!data_) {
        throw std::invalid_argument("[FArray2D::FArray2D] null data pointer given");
    } else if (   (lo.I() > hi.I())
               || (lo.J() > hi.J())
               || (lo.K() > hi.K()) ) {
        std::string   msg =   "[FArray2D::FArray2D] lo ("
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

FArray2D::~FArray2D(void) {
    data_ = nullptr;
}

}

