#include "Milhoja_FArray1D.h"

#include <stdexcept>
#include <string>

namespace milhoja {

//----- static member function definitions
//! Factory function to build an FArray1D that owns data.
FArray1D   FArray1D::buildScratchArray1D(const int lo, const int hi) {
    Real*  scratch = new Real[hi - lo + 1];
    FArray1D   scratchArray = FArray1D{scratch, lo};
    scratchArray.owner_ = true;
    return scratchArray;
}

//----- member function definitions
/** \brief Construct an FArray1D to wrap an input pointer.
  *
  */
FArray1D::FArray1D(Real* data, const int lo)
    : owner_{false},
      data_{data},
      i0_{lo}
{
#ifndef GRID_ERRCHECK_OFF
    // TODO Add tests for these?
    if (!data_) {
        throw std::invalid_argument("[FArray1D::FArray1D] null data pointer given");
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
 * @param lo - the new lower index 
 */
void  FArray1D::reindex(const int lo) {
    if (owner_) {
        i0_ = lo;
    } else {
        throw std::logic_error("[FArray1D::reindex] Reindexing only allowed for scratch arrays");
    }
}

}

