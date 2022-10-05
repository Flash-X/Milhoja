#ifndef MILHOJA_GRID_CONFIGURATION_AMREX_H__
#define MILHOJA_GRID_CONFIGURATION_AMREX_H__

/**
 * A class for creating and populating a singleton that encapsulates all the
 * information needed to configure the Grid AMReX backend singleton.
 * 
 * Refer to the documentation of this class' base class for more information.
 */

#include "Milhoja.h"
#include "Milhoja_GridConfiguration.h"

#ifndef MILHOJA_AMREX_GRID_BACKEND
#error "This file need not be compiled if the AMReX backend isn't used"
#endif

namespace milhoja {

class GridConfigurationAMReX : public GridConfiguration {
public:
    ~GridConfigurationAMReX(void) { };

    GridConfigurationAMReX(GridConfigurationAMReX&)                  = delete;
    GridConfigurationAMReX(const GridConfigurationAMReX&)            = delete;
    GridConfigurationAMReX(GridConfigurationAMReX&&)                 = delete;
    GridConfigurationAMReX& operator=(GridConfigurationAMReX&)       = delete;
    GridConfigurationAMReX& operator=(const GridConfigurationAMReX&) = delete;
    GridConfigurationAMReX& operator=(GridConfigurationAMReX&&)      = delete;

    void     load(void) const override;

private:
    GridConfigurationAMReX(void);

    // Needed for polymorphic singleton
    friend GridConfiguration& GridConfiguration::instance();

    static bool     loaded_;
};

}

#endif

