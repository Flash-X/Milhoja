/**
 * \file    IO.h
 *
 * A toy version of the FLASH-X IO unit implemented in the form of a Singleton
 * class.  It's intended use is only for constructing system-level tests of the
 * runtime and Grid unit.
 */

#ifndef IO_H__
#define IO_H__

#include <string>

#include <mpi.h>

#include "Grid_REAL.h"
#include "Grid_IntVect.h"
#include "DataItem.h"
#include "FArray3D.h"
#include "FArray4D.h"

#include "Flash.h"

namespace orchestration {

class Io {
public:
    ~Io(void);

    Io(Io&)                  = delete;
    Io(const Io&)            = delete;
    Io(Io&&)                 = delete;
    Io& operator=(Io&)       = delete;
    Io& operator=(const Io&) = delete;
    Io& operator=(Io&&)      = delete;

    static void  instantiate(const MPI_Comm comm, const std::string filename);
    static Io&   instance(void);

    //----- INTEGRAL QUANTITIES
#ifdef MAGP_VAR_C
    static constexpr  int  N_GLOBAL_SUM_PROP = 8;
#else
    static constexpr  int  N_GLOBAL_SUM_PROP = 7;
#endif
    static constexpr  int  N_GLOBAL_SUM = N_GLOBAL_SUM_PROP + NMASS_SCALARS;

    void   computeIntegralQuantitiesByBlock(const int threadIndex,
                                            const orchestration::IntVect& lo,
                                            const orchestration::IntVect& hi,
                                            const orchestration::FArray3D& cellVolumes,
                                            const orchestration::FArray4D& solnData);
    void   computeLocalIntegralQuantities(void);
    void   reduceToGlobalIntegralQuantities(void);
    void   writeIntegralQuantities(const orchestration::Real simTime);

private:
    Io(void);

    static bool             instantiated_;
    static MPI_Comm         globalComm_;
    static std::string      intQuantitiesFile_;

    int    rank_;

    //----- INTEGRAL QUANTITIES
    int                     nIntQuantities_;

    // Internal buffers for storing the intermediate integral quantities
    // at the level of an MPI process and for the entire domain
    orchestration::Real*    localIntQuantities_;
    orchestration::Real*    globalIntQuantities_;

    // Internal buffers for accumulating integral quantities as they are 
    // computed across all blocks managed by the MPI process.
    orchestration::Real*    intQuantities_mass_;
    orchestration::Real*    intQuantities_xmom_;
    orchestration::Real*    intQuantities_ymom_;
    orchestration::Real*    intQuantities_zmom_;
    orchestration::Real*    intQuantities_ener_;
    orchestration::Real*    intQuantities_ke_;
    orchestration::Real*    intQuantities_eint_;
    orchestration::Real*    intQuantities_magp_;
};

}

//----- ORCHESTRATION RUNTIME ACTION ROUTINES
namespace ActionRoutines {
    void   Io_computeIntegralQuantitiesByBlock_tile_cpu(const int tId,
                                                        orchestration::DataItem* dataItem);
}

#endif

