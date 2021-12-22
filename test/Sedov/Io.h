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

#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_DataItem.h>
#include <Milhoja_FArray3D.h>
#include <Milhoja_FArray4D.h>

class Io {
public:
    ~Io(void);

    Io(Io&)                  = delete;
    Io(const Io&)            = delete;
    Io(Io&&)                 = delete;
    Io& operator=(Io&)       = delete;
    Io& operator=(const Io&) = delete;
    Io& operator=(Io&&)      = delete;

    static void  instantiate(const std::string filename,
                             const MPI_Comm comm, const int ioRank);
    static Io&   instance(void);

    //----- INTEGRAL QUANTITIES
    static constexpr  int  N_GLOBAL_SUM_PROP = 7;
    static constexpr  int  N_GLOBAL_SUM = N_GLOBAL_SUM_PROP;

    void   computeIntegralQuantitiesByBlock(const int threadIndex,
                                            const milhoja::IntVect& lo,
                                            const milhoja::IntVect& hi,
                                            const milhoja::FArray3D& cellVolumes,
                                            const milhoja::FArray4D& solnData);
    void   computeLocalIntegralQuantities(void);
    void   reduceToGlobalIntegralQuantities(void);
    void   writeIntegralQuantities(const milhoja::Real simTime);

private:
    Io(void);

    static bool             instantiated_;
    static std::string      intQuantitiesFile_;
    static MPI_Comm         comm_;
    static int              ioRank_;

    int    rank_;

    //----- INTEGRAL QUANTITIES
    int                     nIntQuantities_;

    // Internal buffers for storing the intermediate integral quantities
    // at the level of an MPI process and for the entire domain
    milhoja::Real*    localIntQuantities_;
    milhoja::Real*    globalIntQuantities_;

    // Internal buffers for accumulating integral quantities as they are 
    // computed across all blocks managed by the MPI process.
    milhoja::Real*    intQuantities_mass_;
    milhoja::Real*    intQuantities_xmom_;
    milhoja::Real*    intQuantities_ymom_;
    milhoja::Real*    intQuantities_zmom_;
    milhoja::Real*    intQuantities_ener_;
    milhoja::Real*    intQuantities_ke_;
    milhoja::Real*    intQuantities_eint_;
};

//----- ORCHESTRATION RUNTIME ACTION ROUTINES
namespace ActionRoutines {
    void   Io_computeIntegralQuantitiesByBlock_tile_cpu(const int tId,
                                                        milhoja::DataItem* dataItem);
}

#endif

