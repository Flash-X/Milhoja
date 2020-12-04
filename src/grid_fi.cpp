#include "Grid.h"

#include <stdexcept>

#include "Grid_RealVect.h"
#include "Grid_Axis.h"
#include "Flash.h"
#include "actionRoutine.h"
#include "errorEstBlank.h"

using namespace orchestration;

extern "C" {

    /**
     *
     */
    void   grid_init_fi(void) {
        Grid::instantiate();
    }

    /**
     *
     */
    void   grid_init_domain_fi(ACTION_ROUTINE initBlock) {
        // TODO: What to do with error estimate routine?
        //       Should we allow users to supply their own Fortran-based
        //       routine?  Should we only have a handful of C++-based routines
        //       and allow users to specify which through the Fortran interface?
        Grid::instance().initDomain(initBlock, Simulation::errorEstBlank);
    }

    /**
     *
     */
    void   grid_get_domain_bound_box_fi(double lo[NDIM],
                                        double hi[NDIM]) {
        Grid&   grid = Grid::instance();
        RealVect probLo = grid.getProbLo();
        RealVect probHi = grid.getProbHi();

        for (unsigned int i=0; i<NDIM; ++i) {
            lo[i] = static_cast<double>(probLo[i]);
            hi[i] = static_cast<double>(probHi[i]);
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
     * TODO remove this, as FLASH already has a library-agnostic way of doing this.
     * @param vols Pointer to a fortran array of shape (loVect(IAXIS):hiVect(IAXIS),
     *             loVect(JAXIS):hiVect(JAXIS), loVect(KAXIS):hiVect(KAXIS)).
     */
    void   grid_get_cellvolumes_fi(const int level, const int loVect[3], const int hiVect[3], double* vols) {
        Grid&   grid = Grid::instance();
        IntVect lo{LIST_NDIM(loVect[0],loVect[1],loVect[2])};
        IntVect hi{LIST_NDIM(hiVect[0],hiVect[1],hiVect[2])};

#ifndef GRID_ERRCHECK_OFF
        if( NDIM<2 && hiVect[Axis::J]!=loVect[Axis::J] )
            throw std::logic_error("grid_get_cellvolumes_fi: for NDIM<2, lo[JAXIS] must equal hi[JAXIS]");
        if( NDIM<3 && hiVect[Axis::K]!=loVect[Axis::K] )
            throw std::logic_error("grid_get_cellvolumes_fi: for NDIM<3, lo[KAXIS] must equal hi[KAXIS]");
#endif

        grid.fillCellVolumes(static_cast<unsigned int>(level),
                                 lo, hi, vols);


    }

    /**
     *
     */
    void   grid_finalize_fi(void) {
        // The Grid singleton will be destroyed once it goes out of scope
        // Its destructor is reponsible for finalization.
    }
}

