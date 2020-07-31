#include "FArray4D.h"

#include <stdexcept>

namespace orchestration {

FArray4D::FArray4D(Real* data, 
                   const IntVect& begin, const IntVect& end,
                   const unsigned int ncomp)
    : data_{data},
      i0_{begin[Axis::I]},
      j0_{begin[Axis::J]},
//      k0_{begin[Axis::K]},
      jstride_(end[Axis::I] - i0_ + 1),
//      kstride_(jstride_*(end[Axis::J] - j0_ + 1)),
//      nstride_(kstride_*(end[Axis::K] - k0_ + 1)),
      nstride_(jstride_*(end[Axis::J] - j0_ + 1)),
      ncomp_{ncomp}
{
#ifndef GRID_ERRCHECK_OFF
    // TODO Add tests for these?
    if (!data_) {
        throw std::invalid_argument("[FArray4D::FArray4D] null data pointer given");
    } else if (ncomp_ == 0) {
        throw std::invalid_argument("[FArray4D::FArray4D] empty array specified");
    } else if (   (begin[Axis::I] > end[Axis::I])
               || (begin[Axis::J] > end[Axis::J]) ) {
//               || (begin[Axis::K] > end[Axis::K]) ) {
        std::string   msg =   "[FArray4D::FArray4D] begin ("
                            + std::to_string(begin[Axis::I]) + ", "
                            + std::to_string(begin[Axis::J]) + ") "
//                            + std::to_string(begin[Axis::K]) + ") "
                            + " not compatible with end ("
                            + std::to_string(begin[Axis::I]) + ", "
                            + std::to_string(begin[Axis::J]) + ")";
//                            + std::to_string(begin[Axis::K]) + ") "
        throw std::invalid_argument(msg);
    }
#endif
}

FArray4D::~FArray4D(void) {
    // Objects do *not* deallocate resources
    data_ = nullptr;
}

}

