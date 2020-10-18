/**
 * \file    actionRoutine.h
 *
 * Define the interface that must be used for all functions that will be given
 * to a thread team as an action to be applied to all data items enqueued with
 * the team during the associated execution cycle.
 *
 * Note that the only central parameter is the data item on which to apply
 * the action for each invocation of the function.
 */

#ifndef ACTION_ROUTINE_H__
#define ACTION_ROUTINE_H__

#include <memory>
#include "Grid_REAL.h"
#include "Tile.h"

namespace orchestration {

// The data item must be passed as a pointer for interoperability with Fortran
// interface
// TODO: When the dust settles, determine if it is acceptable to send the void*
//       without also sending information to know that a subsequent
//       reinterpret_cast is valid.
using ACTION_ROUTINE = void (*)(const int tId, void* work);


// Error routines calculate a single real value for the error of a Tile.
using ERROR_ROUTINE = Real (*) (std::shared_ptr<Tile> tileDesc, const int iref, const Real ref_filter);

}

#endif

