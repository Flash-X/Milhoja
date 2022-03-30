#ifndef MILHOJA_GRID_CONFIGURATION_H__
#define MILHOJA_GRID_CONFIGURATION_H__

/**
 * An abstract base class for creating and populating a singleton that
 * encapsulates all the information needed to configure the Grid singleton.
 * Only one singleton can be created during program execution and no access to
 * the singleton is granted after it has been cleared.
 *
 * For each Grid backend, there is an associated concrete, subclass of this
 * GridConfiguration.  Upon accessing the GridConfiguration singleton, the
 * appropriate concrete implementation is automatically instantiated.
 * 
 * Before accessing the Grid singleton, it is intended that calling code
 *   - access the GridConfiguration singleton,
 *   - populate the GridConfiguration singleton with configuration values, and
 *   - call the GridConfiguration singleton's load member function.
 *
 * When the Grid singleton is accessed, the Grid singleton configures itself
 * automatically using the contents of this configuration singleton.  After
 * consuming the contents, the Grid singleton clears the configuration
 * singleton.  Therefore, no code should use the configuration singleton after
 * Grid initialization.  Indeed, it is recommended that the creation and
 * population of this singleton be included in a curly brace block so that
 * application code outside the block cannot access the singleton.
 *
 * \todo Add in boundary conditions
 */

#include <mpi.h>

#include "Milhoja_real.h"
#include "Milhoja_coordinateSystem.h"
#include "Milhoja_actionRoutine.h"

namespace milhoja {

class GridConfiguration {
public:
    GridConfiguration(GridConfiguration&)                  = delete;
    GridConfiguration(const GridConfiguration&)            = delete;
    GridConfiguration(GridConfiguration&&)                 = delete;
    GridConfiguration& operator=(GridConfiguration&)       = delete;
    GridConfiguration& operator=(const GridConfiguration&) = delete;
    GridConfiguration& operator=(GridConfiguration&&)      = delete;

    static GridConfiguration&    instance(void);
    virtual void                 load(void) const = 0;
    void                         clear(void);

    // Specification of Problem Domain
    milhoja::CoordSys               coordSys;
    milhoja::Real                   xMin, xMax;
    milhoja::Real                   yMin, yMax;
    milhoja::Real                   zMin, zMax;

    // Specification of Problem Physical Variables
    unsigned int                    nCcVars;

    // Boundary Conditions

    // Specification of Domain Decomposition
    unsigned int                    nGuard;
    unsigned int                    nxb, nyb, nzb; 
    unsigned int                    nBlocksX, nBlocksY, nBlocksZ; 

    // Adaptive Mesh Refinement
    unsigned int                    maxFinestLevel;
    milhoja::ERROR_ROUTINE          errorEstimation;

    // MPI
    MPI_Comm                        mpiComm;

protected:
    GridConfiguration(void);
    virtual ~GridConfiguration(void) { };

    bool    isValid(void) const;

    static bool    cleared_;  //< True if clear() has been called.
};

}

#endif

