#include "Grid.h"

#include <stdexcept>

#include "Grid_Vector.h"
#include "Flash.h"
#include "runtimeTask.h"

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
                               TASK_FCN initBlock) {
        Grid&   grid = Grid::instance();
        grid.initDomain(static_cast<grid::Real>(xMin),
                        static_cast<grid::Real>(xMax),
                        static_cast<grid::Real>(yMin),
                        static_cast<grid::Real>(yMax),
                        static_cast<grid::Real>(zMin),
                        static_cast<grid::Real>(zMax),
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
        grid::Vector<grid::Real> domainLo = grid.getDomainLo();
        grid::Vector<grid::Real> domainHi = grid.getDomainLo();

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
        grid::Vector<grid::Real> deltas = grid.getDeltas(static_cast<unsigned int>(level));

        for (unsigned int i=0; i<NDIM; ++i) {
            deltas[i] = static_cast<double>(deltas[i]);
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

