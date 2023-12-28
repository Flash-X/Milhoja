#include "Milhoja_GridConfiguration.h"

#include <stdexcept>

#include "Milhoja.h"
#include "Milhoja_Logger.h"
#include "Milhoja_axis.h"

#ifdef MILHOJA_AMREX_GRID_BACKEND
#include "Milhoja_GridConfigurationAMReX.h"
#endif

namespace milhoja {

//----- STATIC DATA MEMBER INITIALIZATION
bool   GridConfiguration::cleared_ = false;

//----- STATIC MEMBER FUNCTION DEFINITIONS
/**
 * Obtain access to the configuration singleton.  It is a logic error to call
 * this member function after clear() has been called.
 *
 * Configuration values are set to nonsensical values the first time that
 * this function is called.
 */
GridConfiguration&   GridConfiguration::instance(void) {
    if (cleared_) {
        throw std::logic_error("[GridConfiguration::instance] Configuration already consumed");
    }

#ifdef MILHOJA_AMREX_GRID_BACKEND
    static GridConfigurationAMReX   singleton;
#else
#error "Need to specify Grid implementation with MILHOJA_[NAME]_GRID_BACKEND macro"
#endif

    return singleton;
}

//----- MEMBER FUNCTION DEFINITIONS
/**
 * Determine if the current configuration values are sensible.  These checks are
 * sanity checks and do *not* perform any checks that could be library-specific.
 *
 * \todo Confirm that function pointers are not nullptr
 * \todo Confirm that thread counts are positive
 * \todo What is the requirement on the Min/Max values above MILHOJA_NDIM?  Do we need
 *       adjust these tests in accord with this?
 */
bool GridConfiguration::isValid(void) const {
    bool    isValid = true;

#ifdef FULL_MILHOJAGRID
    if ((xMax <= xMin) || (yMax <= yMin) || (zMax <= zMin)) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Invalid physical domain");
        isValid = false;
    } else if (nCcVars <= 0) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - nCcVars not positive");
        isValid = false;
    } else if ((nxb <= 0) || (nyb <= 0) || (nzb <= 0)) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Invalid block size");
        isValid = false;
    } else if ((nBlocksX <= 0) || (nBlocksY <= 0) || (nBlocksZ <= 0)) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Invalid domain block array");
        isValid = false;
    } else if (!errorEstimation) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Null errorEstimation given");
        isValid = false;
    } else if (mpiComm == MPI_COMM_NULL) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Null MPI communicator");
        isValid = false;
    } else if (   (coordSys != CoordSys::Cartesian)
               && (coordSys != CoordSys::Cylindrical)
               && (coordSys != CoordSys::Spherical)) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Invalid coordinate system");
        isValid = false;
    } else if (   ((loBCs[Axis::I] != BCs::Periodic) && (loBCs[Axis::I] != BCs::External))
               || ((hiBCs[Axis::I] != BCs::Periodic) && (hiBCs[Axis::I] != BCs::External))
#if MILHOJA_NDIM >= 2
               || ((loBCs[Axis::J] != BCs::Periodic) && (loBCs[Axis::J] != BCs::External))
               || ((hiBCs[Axis::J] != BCs::Periodic) && (hiBCs[Axis::J] != BCs::External))
#endif
#if MILHOJA_NDIM == 3
               || ((loBCs[Axis::K] != BCs::Periodic) && (loBCs[Axis::K] != BCs::External))
               || ((hiBCs[Axis::K] != BCs::Periodic) && (hiBCs[Axis::K] != BCs::External))
#endif
    ) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Invalid BCs");
        isValid = false;
    } else if (   ((loBCs[Axis::I] == BCs::Periodic) && (hiBCs[Axis::I] != BCs::Periodic))
               || ((loBCs[Axis::I] != BCs::Periodic) && (hiBCs[Axis::I] == BCs::Periodic)) ) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Mixed periodic BC in X");
        isValid = false;
#if MILHOJA_NDIM >= 2
    } else if (   ((loBCs[Axis::J] == BCs::Periodic) && (hiBCs[Axis::J] != BCs::Periodic))
               || ((loBCs[Axis::J] != BCs::Periodic) && (hiBCs[Axis::J] == BCs::Periodic)) ) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Mixed periodic BC in Y");
        isValid = false;
#endif
#if MILHOJA_NDIM == 3
    } else if (   ((loBCs[Axis::K] == BCs::Periodic) && (hiBCs[Axis::K] != BCs::Periodic))
               || ((loBCs[Axis::K] != BCs::Periodic) && (hiBCs[Axis::K] == BCs::Periodic)) ) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Mixed periodic BC in Z");
        isValid = false;
#endif
    } else if (   (!externalBcRoutine)
               && (   (loBCs[Axis::I] != BCs::Periodic) || (hiBCs[Axis::I] != BCs::Periodic)
#if MILHOJA_NDIM >= 2
                   || (loBCs[Axis::J] != BCs::Periodic) || (hiBCs[Axis::J] != BCs::Periodic)
#endif
#if MILHOJA_NDIM == 3
                   || (loBCs[Axis::K] != BCs::Periodic) || (hiBCs[Axis::K] != BCs::Periodic)
#endif
                  ) ) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - null BC routine");
        isValid = false;
    }

#ifdef MILHOJA_AMREX_GRID_BACKEND
    if (   (ccInterpolator != Interpolator::CellConservativeLinear)
        && (ccInterpolator != Interpolator::CellConservativeProtected)
        && (ccInterpolator != Interpolator::CellConservativeQuartic)
        && (ccInterpolator != Interpolator::CellPiecewiseConstant)
        && (ccInterpolator != Interpolator::CellBilinear)
        && (ccInterpolator != Interpolator::CellQuadratic)) {
        Logger::instance().log("[GridConfiguration::isValid] ERROR - Invalid AMReX CC Interpolator");
        isValid = false;
   }
#endif
#endif

    return isValid;
}

/**
 * Set nonsensical configuration values.  It is a logic error to call this
 * function more than once.
 */
void GridConfiguration::clear(void) {
    if (cleared_) {
        throw std::logic_error("[GridConfiguration::clear] Configuration already cleared");
    }

    coordSys          = CoordSys::Cartesian;
    xMin              =  1.0;
    xMax              =  0.0;
    yMin              =  1.0;
    yMax              =  0.0;
    zMin              =  1.0;
    zMax              =  0.0;
    loBCs[Axis::I]    = BCs::External;
    hiBCs[Axis::I]    = BCs::Periodic;
    loBCs[Axis::J]    = BCs::Periodic;
    hiBCs[Axis::J]    = BCs::Periodic;
    loBCs[Axis::K]    = BCs::Periodic;
    hiBCs[Axis::K]    = BCs::Periodic;
    externalBcRoutine = nullptr;
    nCcVars           =  0;
    nFluxVars         =  0;
    nxb               =  0; 
    nyb               =  0; 
    nzb               =  0; 
    nGuard            =  0;
    nBlocksX          =  0; 
    nBlocksY          =  0; 
    nBlocksZ          =  0; 
    maxFinestLevel    =  0;
    errorEstimation   = nullptr;
    ccInterpolator    = Interpolator::CellConservativeLinear;
    mpiComm           = MPI_COMM_NULL;

    // Limit possiblity that calling code can access the
    // configuration at a later time.
    cleared_ = true;
}

/**
 * Set nonsensical configuration values and leave the singleton is an
 * accessible state.
 */
GridConfiguration::GridConfiguration(void) {
    // Set intentially bad values
    clear();

    // The configuration was not consumed here
    cleared_ = false;
}

}

