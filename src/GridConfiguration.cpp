#include "GridConfiguration.h"

#include <stdexcept>

#include "OrchestrationLogger.h"

namespace orchestration {

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

    static GridConfiguration    singleton;
    return singleton;
}

//----- MEMBER FUNCTION DEFINITIONS
/**
 * Determine if the current configuration values are sensible.  These checks are
 * sanity checks and do *not* perform any checks that could be library-specific.
 *
 * \todo Confirm that function pointers are not nullptr
 * \todo Confirm that thread counts are positive
 * \todo What is the requirement on the Min/Max values above NDIM?  Do we need
 *       adjust these tests in accord with this?
 */
bool GridConfiguration::isValid(void) const {
    bool    isValid = true;

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
    }

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

    xMin                     =  1.0;
    xMax                     =  0.0;
    yMin                     =  1.0;
    yMax                     =  0.0;
    zMin                     =  1.0;
    zMax                     =  0.0;
    nCcVars                  =  0;
    initBlock                = nullptr;
    nCpuThreads_init         =  0;
    nDistributorThreads_init =  0;
    nxb                      =  0; 
    nyb                      =  0; 
    nzb                      =  0; 
    nGuard                   =  0;
    nBlocksX                 =  0; 
    nBlocksY                 =  0; 
    nBlocksZ                 =  0; 
    maxFinestLevel           =  0;
    errorEstimation          = nullptr;

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

