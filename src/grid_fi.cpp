#include "Grid.h"

#include <stdexcept>

#include "Flash.h"
#include "runtimeTask.h"

extern "C" {

    /**
     *
     */
    void   grid_init_fi(void) {
        Grid<NXB,NYB,NZB,NGUARD>::instance();
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
                               TASK_FCN<Tile> initBlock) {
        Grid<NXB,NYB,NZB,NGUARD>*   grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
        grid->initDomain(static_cast<amrex::Real>(xMin),
                         static_cast<amrex::Real>(xMax),
                         static_cast<amrex::Real>(yMin),
                         static_cast<amrex::Real>(yMax),
                         static_cast<amrex::Real>(zMin),
                         static_cast<amrex::Real>(zMax),
                         static_cast<unsigned int>(nBlocksX),
                         static_cast<unsigned int>(nBlocksY),
                         static_cast<unsigned int>(nBlocksZ),
                         static_cast<unsigned int>(nVars),
                         initBlock);
        grid = nullptr;
    }

    /**
     *
     */
    void   grid_get_domain_bound_box_fi(double lo[AMREX_SPACEDIM], 
                                        double hi[AMREX_SPACEDIM]) {
        Grid<NXB,NYB,NZB,NGUARD>*   grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
        amrex::Geometry&  geometry = grid->geometry();
        grid = nullptr;

        for (unsigned int i=0; i<AMREX_SPACEDIM; ++i) {
            lo[i] = static_cast<double>(geometry.ProbLo(i));
            hi[i] = static_cast<double>(geometry.ProbHi(i));
        }
    }

    /**
     *
     */
    void   grid_get_deltas_fi(const int level, double deltas[AMREX_SPACEDIM]) {
        Grid<NXB,NYB,NZB,NGUARD>*   grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
        amrex::Geometry&  geometry = grid->geometry();
        grid = nullptr;

        for (unsigned int i=0; i<AMREX_SPACEDIM; ++i) {
            deltas[i] = static_cast<double>(geometry.CellSize(i));
        }
    }

    /**
     *
     */
    void   grid_finalize_fi(void) {
        delete Grid<NXB,NYB,NZB,NGUARD>::instance();
    }
}

