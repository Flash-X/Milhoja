#include "loadGridConfiguration.h"

#include <Milhoja_GridConfiguration.h>

#include "Sedov.h"
#include "Simulation.h"
#include "errorEstBlank.h"

#include "Flash_par.h"

void loadGridConfiguration(void) {
    milhoja::GridConfiguration&   cfg = milhoja::GridConfiguration::instance();

    cfg.xMin                     = rp_Grid::X_MIN;
    cfg.xMax                     = rp_Grid::X_MAX;
    cfg.yMin                     = rp_Grid::Y_MIN;
    cfg.yMax                     = rp_Grid::Y_MAX;
    cfg.zMin                     = rp_Grid::Z_MIN;
    cfg.zMax                     = rp_Grid::Z_MAX;
    cfg.nxb                      = rp_Grid::NXB;
    cfg.nyb                      = rp_Grid::NYB;
    cfg.nzb                      = rp_Grid::NZB;
    cfg.nCcVars                  = NUNKVAR;
    cfg.nGuard                   = NGUARD;
    cfg.nBlocksX                 = rp_Grid::N_BLOCKS_X;
    cfg.nBlocksY                 = rp_Grid::N_BLOCKS_Y;
    cfg.nBlocksZ                 = rp_Grid::N_BLOCKS_Z;
    cfg.maxFinestLevel           = rp_Grid::LREFINE_MAX;
    cfg.initBlock                = Simulation::setInitialConditions_tile_cpu;
    cfg.nDistributorThreads_init = rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC;
    cfg.nCpuThreads_init         = rp_Simulation::N_THREADS_FOR_IC;
    cfg.errorEstimation          = Simulation::errorEstBlank;

    cfg.load();
}

