#include "Milhoja_GridConfigurationAMReX.h"

#include <stdexcept>

#include <AMReX_CoordSys.H>
#include <AMReX_ParmParse.H>

#include "Milhoja.h"
#include "Milhoja_Logger.h"

namespace milhoja {

//----- STATIC DATA MEMBER INITIALIZATION
bool   GridConfigurationAMReX::loaded_ = false;

//----- MEMBER FUNCTION DEFINITIONS
/**
 * Construct the singleton.
 */
GridConfigurationAMReX::GridConfigurationAMReX(void)
    : GridConfiguration()
{ }

/**
 * Load into AMReX the current configuration values stored in the object.  This
 * routine does *not* clear these values once consumed as GridAMReX owns some of
 * the configuration values and must have access to them when the Grid singleton
 * is instantiated.  It is, therefore, the responsibility of GridAMReX to clear
 * the contents upon assuming ownership of its values.
 */
void GridConfigurationAMReX::load(void) const {
    if        (cleared_) {
        throw std::logic_error("[GridConfigurationAMReX::load] Configuration already consumed");
    } else if (loaded_) {
        throw std::logic_error("[GridConfigurationAMReX::load] Configuration already loaded");
    } else if (!isValid()) {
        throw std::invalid_argument("[GridConfigurationAMReX::load] Invalid configuration");
    }

    amrex::ParmParse    ppGeo("geometry");

    std::vector<int>   isPeriodic(MILHOJA_MDIM, 0);
    for (auto i=0; i<isPeriodic.size(); ++i) {
        if ((loBCs[i] == BCs::Periodic) && (hiBCs[i] == BCs::Periodic)) {
            isPeriodic[i] = 1;
        } 
    }
    ppGeo.addarr("is_periodic", isPeriodic);

#ifdef FULL_MILHOJAGRID
    switch (coordSys) {
    case CoordSys::Cartesian:
        ppGeo.add("coord_sys", amrex::CoordSys::CoordType::cartesian);
        break;
    case CoordSys::Cylindrical:
        ppGeo.add("coord_sys", amrex::CoordSys::CoordType::RZ);
        break;
    case CoordSys::Spherical:
        ppGeo.add("coord_sys", amrex::CoordSys::CoordType::SPHERICAL);
        break;
    default:
        throw std::invalid_argument("[GridConfigurationAMReX::load] coordSys invalid");
    }
#endif
    ppGeo.addarr("prob_lo", std::vector<Real>{LIST_NDIM(xMin,
                                                        yMin,
                                                        zMin)});
    ppGeo.addarr("prob_hi", std::vector<Real>{LIST_NDIM(xMax,
                                                        yMax,
                                                        zMax)});

    // TODO: Check for overflow
    int  lrefineMax_i = static_cast<int>(maxFinestLevel);
    int  nxb_i        = static_cast<int>(nxb);
    int  nyb_i        = static_cast<int>(nyb);
    int  nzb_i        = static_cast<int>(nzb);
    int  nCellsX_i    = static_cast<int>(nxb * nBlocksX);
    int  nCellsY_i    = static_cast<int>(nyb * nBlocksY);
    int  nCellsZ_i    = static_cast<int>(nzb * nBlocksZ);

#ifdef FULL_MILHOJAGRID
    amrex::ParmParse ppAmr("amr");

    ppAmr.add("v", 0); //verbosity
    //ppAmr.add("regrid_int",nrefs); //how often to refine
    ppAmr.add("max_level", lrefineMax_i - 1); //0-based
    ppAmr.addarr("n_cell", std::vector<int>{LIST_NDIM(nCellsX_i,
                                                      nCellsY_i,
                                                      nCellsZ_i)});

    // This configures AMReX into an effective octree mode.  In particular, it
    // is still running a level-based AMR system.
    ppAmr.add("max_grid_size_x",    nxb_i);
    ppAmr.add("max_grid_size_y",    nyb_i);
    ppAmr.add("max_grid_size_z",    nzb_i);
    ppAmr.add("blocking_factor_x",  nxb_i * 2);
    ppAmr.add("blocking_factor_y",  nyb_i * 2);
    ppAmr.add("blocking_factor_z",  nzb_i * 2);
    ppAmr.add("refine_grid_layout", 0);
    ppAmr.add("grid_eff",           1.0);
    ppAmr.add("n_proper",           1);
    ppAmr.add("n_error_buf",        0);
    ppAmr.addarr("ref_ratio",       std::vector<int>(lrefineMax_i, 2));
#endif

    // DEV_GUIDE: Satisfy GridConfiguration requirements
    int     wasInitialized = 0;
    MPI_Initialized(&wasInitialized);
    if (!wasInitialized) {
        throw std::logic_error("[GridConfigurationAMReX::load] Please initialize MPI first");
    }

    // It appears that AMReX must be initialized before the derived AmrCore
    // class is instantiated.  Therefore, we must perform the initialization of
    // AMReX here.
#ifdef FULL_MILHOJAGRID
    amrex::Initialize(mpiComm);
#endif

    loaded_ = true;

    Logger::instance().log("[GridConfigurationAMReX] Loaded configuration values into AMReX");
}

}

