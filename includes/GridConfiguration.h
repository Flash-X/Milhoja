#ifndef GRID_CONFIGURATION_H__
#define GRID_CONFIGURATION_H__

/**
 * A class for creating and populating a singleton that encapsulates all the
 * information needed to configure the Grid singleton.  Only one singleton can
 * be created during program execution and no access to the singleton is granted
 * after it has been cleared.
 * 
 * It is intended that calling code access the singleton and populate it with
 * configuration values before the Grid singleton is initialized, during which
 * time the Grid singleton configures itself automatically using the contents of
 * this configuration singleton.  After consuming the contents, the Grid
 * singleton clears the configuration singleton.  Thereafter, no code should use
 * the configuration singleton after Grid initialization.  Indeed, it is
 * recommended that the creation and population of this singleton be included in
 * a curly brace block so that application code outside the block cannot access
 * the singleton.
 *
 * \todo Add in coordinate system
 * \todo Add in boundary conditions
 */

#include "milhoja.h"
#include "Grid_REAL.h"
#include "actionRoutine.h"

namespace orchestration {

class GridConfiguration {
public:
    GridConfiguration(GridConfiguration&)                  = delete;
    GridConfiguration(const GridConfiguration&)            = delete;
    GridConfiguration(GridConfiguration&&)                 = delete;
    GridConfiguration& operator=(GridConfiguration&)       = delete;
    GridConfiguration& operator=(const GridConfiguration&) = delete;
    GridConfiguration& operator=(GridConfiguration&&)      = delete;

    static GridConfiguration&    instance(void);
    bool                         isValid(void) const;
    void                         clear(void);

    // Specification of Problem Domain
    orchestration::Real             xMin, xMax;
    orchestration::Real             yMin, yMax;
    orchestration::Real             zMin, zMax;

    // Specification of Problem Physical Variables
    unsigned int                    nCcVars;

    // Initial Conditions
    orchestration::ACTION_ROUTINE   initBlock;
    unsigned int                    nCpuThreads_init;
    unsigned int                    nDistributorThreads_init;

    // Boundary Conditions

    // Specification of Domain Decomposition
    unsigned int                    nGuard;
    unsigned int                    nxb, nyb, nzb; 
    unsigned int                    nBlocksX, nBlocksY, nBlocksZ; 

    // Adaptive Mesh Refinement
    unsigned int                    maxFinestLevel;
    orchestration::ERROR_ROUTINE    errorEstimation;

private:
    GridConfiguration(void);
    ~GridConfiguration(void) { };

    static bool    cleared_;  //< True if clear() has been called.
};

}

#endif

