#include "Grid.h"

#include <stdexcept>

#include "Grid_RealVect.h"
#include "Flash.h"
#include "actionRoutine.h"

using namespace orchestration;

extern "C" {

    /**
     *
     */
    void   grid_init_fi(void) {
        Grid::instance();
    }

    /**
     *
     */
    void   grid_init_domain_fi(const double xMin, const double xMax,
                               const double yMin, const double yMax,
                               const double zMin, const double zMax,
                               const int nBlocksX,
                               const int nBlocksY,
                               const int nBlocksZ,
                               const int nVars,
                               ACTION_ROUTINE initBlock) {
        Grid&   grid = Grid::instance();
        grid.initDomain(static_cast<Real>(xMin),
                        static_cast<Real>(xMax),
                        static_cast<Real>(yMin),
                        static_cast<Real>(yMax),
                        static_cast<Real>(zMin),
                        static_cast<Real>(zMax),
                        static_cast<unsigned int>(nBlocksX),
                        static_cast<unsigned int>(nBlocksY),
                        static_cast<unsigned int>(nBlocksZ),
                        static_cast<unsigned int>(nVars),
                        initBlock);
    }

    /**
     *
     */
    void   grid_get_domain_bound_box_fi(double lo[NDIM],
                                        double hi[NDIM]) {
        Grid&   grid = Grid::instance();
        RealVect domainLo = grid.getDomainLo();
        RealVect domainHi = grid.getDomainLo();

        for (unsigned int i=0; i<NDIM; ++i) {
            lo[i] = static_cast<double>(domainLo[i]);
            hi[i] = static_cast<double>(domainHi[i]);
        }
    }

    /**
     *
     */
    void   grid_get_deltas_fi(const int level, double deltas[NDIM]) {
        Grid&   grid = Grid::instance();
        RealVect del = grid.getDeltas(static_cast<unsigned int>(level));

        for (unsigned int i=0; i<NDIM; ++i) {
            deltas[i] = static_cast<double>(del[i]);
        }
    }

    /**
     *
     */
    void   grid_finalize_fi(void) {
        // The Grid singleton will be destroyed once it goes out of scope
        // Its destructor is reponsible for finalization.
    }
}

