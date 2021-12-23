/**
 * \file    Milhoja_actionRoutine.h
 *
 * Define the interface that must be used for all functions that will be given
 * to a thread team as an action to be applied to all data items enqueued with
 * the team during the associated execution cycle.
 *
 * Note that the only central parameter is the data item on which to apply
 * the action for each invocation of the function.
 */

#ifndef MILHOJA_ACTION_ROUTINE_H__
#define MILHOJA_ACTION_ROUTINE_H__

#include <memory>

#include "Milhoja_DataItem.h"

namespace milhoja {

// The data item must be passed as a pointer for interoperability with Fortran
// interface
using ACTION_ROUTINE = void (*)(const int tId, DataItem* work);


// TODO rethink how error calculation/tagging for refinement will work.
class Tile;
using ERROR_ROUTINE = void (*) (std::shared_ptr<Tile> tileDesc, int* tptr);

}

#endif

