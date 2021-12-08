#include "Io.h"

#include <mpi.h>

#include <stdexcept>

#include "milhoja.h"
#include "Tile.h"
#include "Grid.h"
#include "OrchestrationLogger.h"

#include "constants.h"
#include "Flash_par.h"

using namespace orchestration;

//----- STATIC MEMBER DEFINITIONS
bool             Io::instantiated_ = false;
std::string      Io::intQuantitiesFile_ = "";
MPI_Comm         Io::comm_ = MPI_COMM_NULL;
int              Io::ioRank_ = -1;

/**
 * Instantiate the IO unit singleton and allow it to initialize the unit.
 */
void   Io::instantiate(const std::string filename,
                       const MPI_Comm comm, const int ioRank) {
    Logger::instance().log("[IO] Initializing...");

    if (instantiated_) {
        throw std::logic_error("[Io::instantiate] Io is already instantiated");
    } else if (filename == "") {
        throw std::invalid_argument("[Io::instantiate] Empty filename given");
    } else if (ioRank < 0) {
        throw std::invalid_argument("[Io::instantiate] Negative MPI rank");
    }
    intQuantitiesFile_ = filename;
    comm_ = comm;
    ioRank_ = ioRank;

    instantiated_ = true;
    Io::instance();

    Logger::instance().log("[IO] Created and ready for use");
}

/**
 * Obtain access to the IO unit singleton object.
 *
 * @return The object
 */
Io&   Io::instance(void) {
    if (!instantiated_) {
        throw std::logic_error("[Io::instance] Instantiate Io first");
    }
    static Io  singleton;
    return singleton;
}

//----- STANDARD DEFINITIONS
/**
 * Construct and initialize the IO unit.
 *
 * \todo Setup the number of integral quantities so that they can incorporate
 *       mass scalar quantities if so desired (c.f. io_writeMscalarIntegrals).
 *
 * \todo Include the possibility of starting a simulation as a restart.  Look at
 *       FLASH-X to see how they manange files like the integral quantities output.
 */
Io::Io(void)
    : rank_{-1},
      nIntQuantities_{N_GLOBAL_SUM_PROP},
      localIntQuantities_{new Real[nIntQuantities_]},
      globalIntQuantities_{nullptr},
      intQuantities_mass_{nullptr},
      intQuantities_xmom_{nullptr}, 
      intQuantities_ymom_{nullptr}, 
      intQuantities_zmom_{nullptr}, 
      intQuantities_ener_{nullptr}, 
      intQuantities_ke_{nullptr}, 
      intQuantities_eint_{nullptr}, 
      intQuantities_magp_{nullptr} 
{
    Logger&   logger = Logger::instance();

    if (MPI_Comm_rank(comm_, &rank_) != MPI_SUCCESS) {
        throw std::runtime_error("[Io::Io] Unable to acquire MPI rank");
    }

    // Construct buffer that receives final integral quantity results
    if (rank_ == ioRank_) {
        globalIntQuantities_ = new Real[nIntQuantities_];
    }

    // Construct buffers that serve as accumulators of intermediate integral
    // quantity results as computed on a per-block basis
    //
    // Leave the accumulators zeroed so that they are ready for use.
    logger.log("[IO] Integral quantities to be computed and output are");
    unsigned int   nThreads = rp_Runtime::N_THREADS_PER_TEAM;
#ifdef DENS_VAR_C
    intQuantities_mass_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_mass_[i] = 0.0_wp;
    }
    logger.log("[IO]     Mass");
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
    intQuantities_xmom_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_xmom_[i] = 0.0_wp;
    }
    logger.log("[IO]     X-Momentum");
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
    intQuantities_ymom_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_ymom_[i] = 0.0_wp;
    }
    logger.log("[IO]     Y-Momentum");
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
    intQuantities_zmom_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_zmom_[i] = 0.0_wp;
    }
    logger.log("[IO]     Z-Momentum");
#endif
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
    intQuantities_ener_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_ener_[i] = 0.0_wp;
    }
    logger.log("[IO]     Total Energy");
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
    intQuantities_ke_   = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_ke_[i] = 0.0_wp;
    }
    logger.log("[IO]     Kinetic Energy");
#endif
#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
    intQuantities_eint_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_eint_[i] = 0.0_wp;
    }
    logger.log("[IO]     Internal Energy");
#endif
#ifdef MAGP_VAR_C
    intQuantities_magp_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_magp_[i] = 0.0_wp;
    }
    logger.log("[IO]     Magnetic Energy");
#endif

    // Append integral quantity header to file
    if (rank_ == ioRank_) {
        FILE*   fptr = fopen(intQuantitiesFile_.c_str(), "a");
        if (!fptr) {
            std::string msg = "[Io::Io] Unable to open integral quantities output file "
                              + intQuantitiesFile_;
            throw std::runtime_error(msg);
        }

#ifdef MAGP_VAR_C
        fprintf(fptr, "#%24s %25s %25s %25s %25s %25s %25s %25s %25s\n",
                    "time", 
                    "mass",
                    "x-momentum",
                    "y-momentum",
                    "z-momentum",
                    "E_total",
                    "E_kinetic",
                    "E_internal",
                    "MagEnergy");
#else
        fprintf(fptr, "#%24s %25s %25s %25s %25s %25s %25s %25s\n",
                    "time", 
                    "mass",
                    "x-momentum",
                    "y-momentum",
                    "z-momentum",
                    "E_total",
                    "E_kinetic",
                    "E_internal");
#endif
        fclose(fptr);
        fptr = NULL;
    }
    logger.log(  "[IO] Header appended to integral quantity output file "
               + intQuantitiesFile_);
}

/**
 * Clean-up the IO unit.
 */
Io::~Io(void) {
    Logger::instance().log("[IO] Finalizing...");

    if (intQuantities_mass_) {
        delete [] intQuantities_mass_;
        intQuantities_mass_ = nullptr;
    }

    if (intQuantities_xmom_) {
        delete [] intQuantities_xmom_;
        intQuantities_xmom_ = nullptr;
    }

    if (intQuantities_ymom_) {
        delete [] intQuantities_ymom_;
        intQuantities_ymom_ = nullptr;
    }

    if (intQuantities_zmom_) {
        delete [] intQuantities_zmom_;
        intQuantities_zmom_ = nullptr;
    }

    if (intQuantities_ener_) {
        delete [] intQuantities_ener_;
        intQuantities_ener_ = nullptr;
    }

    if (intQuantities_ke_) {
        delete [] intQuantities_ke_;
        intQuantities_ke_ = nullptr;
    }

    if (intQuantities_eint_) {
        delete [] intQuantities_eint_;
        intQuantities_eint_ = nullptr;
    }

    if (intQuantities_magp_) {
        delete [] intQuantities_magp_;
        intQuantities_magp_ = nullptr;
    }

    if (localIntQuantities_) {
        delete [] localIntQuantities_;
        localIntQuantities_ = nullptr;
    }

    if (globalIntQuantities_) {
        delete [] globalIntQuantities_;
        globalIntQuantities_ = nullptr;
    }

    Logger::instance().log("[IO] Finalized");

    intQuantitiesFile_ = "";
    comm_ = MPI_COMM_NULL;
    ioRank_ = -1;
    instantiated_ = false;
}

void   Io::computeLocalIntegralQuantities(void) {
    orchestration::Grid&     grid    = orchestration::Grid::instance();

    unsigned int            level{0};
    std::shared_ptr<Tile>   tileDesc{};
    for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
        tileDesc = ti->buildCurrentTile();

        unsigned int        level = tileDesc->level();
        const IntVect       lo    = tileDesc->lo();
        const IntVect       hi    = tileDesc->hi();
        const FArray4D      U     = tileDesc->data();

        Real   volumes_buffer[  (hi.I() - lo.I() + 1)
                              * (hi.J() - lo.J() + 1)
                              * (hi.K() - lo.K() + 1)];
        grid.fillCellVolumes(level, lo, hi, volumes_buffer); 
        const FArray3D   volumes{volumes_buffer, lo, hi};

        // FIXME: This is an inefficient hack whereby we can use the present
        // ThreadTeam reduction trick but have the integral quantities
        // aggregated into a single variable.  This hack should disappear once
        // the reduction is built into calls to the runtime itself.
        computeIntegralQuantitiesByBlock(0, lo, hi, volumes, U);
    }
}

/**
 * Integrate the physical integral quantities of interest across the given
 * region.  Note that this function
 *  - is not intended to be applied to any arbitrary rectangular region of
 *    the domain, but rather to blocks,
 *  - is intended for computation across the interior of the given block, and
 *  - is intended for use with leaf blocks.
 *
 * The integrated quantities for the given block are accumulated into internal
 * arrays that are indexed by the ID of the thread in the team that applied this
 * action to the given block.  Since any given thread in a team can apply its
 * action to at most one data item at a time, this indexing scheme prevents race
 * conditions on these internal arrays and therefore allows for multithreaded
 * computation of the quantities across all blocks managed by the process.
 *
 * The internal buffers must be zeroed before using this function.  The buffers
 * are zeroed by default at instantiation and can otherwise be zeroed by calling
 * the function reduceToGlobalIntegralQuantities.
 *
 * \param threadIdx - the index into the array in which this result should
 *                    accumulate.  This value must be in the set {0, 1, ..., N-1},
 *                    where N is the total number of threads in the thread team used
 *                    to apply this action.
 * \param lo - the lo corner of the block's interior
 * \param hi - the hi corner of the block's interior
 * \param cellVolumes - the volumes of each cell in the block
 * \param solnData - the data to integrate
 */
void   Io::computeIntegralQuantitiesByBlock(const int threadIdx,
                                            const orchestration::IntVect& lo,
                                            const orchestration::IntVect& hi,
                                            const orchestration::FArray3D& cellVolumes,
                                            const orchestration::FArray4D& solnData) {
    Real    dvol = 0.0_wp;
#if defined(DENS_VAR_C)
    Real    mass = 0.0_wp;
    Real    massSum = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
    Real    xmomSum = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
    Real    ymomSum = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
    Real    zmomSum = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
    Real    enerSum = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
    Real    keSum   = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
    Real    eintSum = 0.0_wp;
#endif
#ifdef MAGP_VAR_C
    Real    magpSum = 0.0_wp;
#endif

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                dvol = cellVolumes(i, j, k);

#if defined(DENS_VAR_C)
                // mass
                mass = solnData(i, j, k, DENS_VAR_C) * dvol;
                massSum += mass;
#endif

                // momentum
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
                xmomSum += mass * solnData(i, j, k, VELX_VAR_C);
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
                ymomSum += mass * solnData(i, j, k, VELY_VAR_C);
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
                zmomSum += mass * solnData(i, j, k, VELZ_VAR_C);
#endif

                // total energy
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
                enerSum += mass * solnData(i, j, k, ENER_VAR_C);
#ifdef MAGP_VAR_C
                // total plasma energy
                enerSum += solnData(i, j, k, MAGP_VAR_C) * dvol;
#endif
#endif

#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
                // kinetic energy
                keSum += 0.5_wp * mass
                                * (  std::pow(solnData(i, j, k, VELX_VAR_C), 2)
                                   + std::pow(solnData(i, j, k, VELY_VAR_C), 2)
                                   + std::pow(solnData(i, j, k, VELZ_VAR_C), 2));
#endif

#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
              // internal energy
              eintSum += mass * solnData(i, j, k, EINT_VAR_C);
#endif

#ifdef MAGP_VAR_C
              // magnetic energy
              magpSum += solnData(i, j, k, MAGP_VAR_C) * dvol;
#endif
            }
        }
    }

#ifdef DENS_VAR_C
    intQuantities_mass_[threadIdx] += massSum;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
    intQuantities_xmom_[threadIdx] += xmomSum;
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
    intQuantities_ymom_[threadIdx] += ymomSum;
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
    intQuantities_zmom_[threadIdx] += zmomSum;
#endif
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
    intQuantities_ener_[threadIdx] += enerSum;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
    intQuantities_ke_[threadIdx]   += keSum;
#endif
#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
    intQuantities_eint_[threadIdx] += eintSum;
#endif
#ifdef MAGP_VAR_C
    intQuantities_magp_[threadIdx] += magpSum;
#endif
}

/**
 * Use the integral quantities as pre-computed on a per-block basis by
 * computeIntegralQuantitiesByBlock to compute the same quantities integrated
 * across all blocks managed by the process and then across all processes.
 *
 * The last stage requires a DATA MOVEMENT in the form of an MPI reduction.
 *
 * This function leaves all internal accumulating arrays used by
 * computeIntegralQuantitiesByBlock zeroed so that they are ready for immediate
 * use the next time that integral quantities need to be computed.
 *
 * \todo Don't get nThreadsPerTeam from orchestration?  Where from?
 */
void   Io::reduceToGlobalIntegralQuantities(void) {
    for (unsigned int i=0; i<nIntQuantities_; ++i) {
        localIntQuantities_[i] = 0.0_wp;
    }

    // Any or all of the threads in the thread team that ran
    // computeIntegralQuantitiesByBlock could have computed part of the
    // integration.  Therefore, we integrate across all threads.
    for (unsigned int i=0; i<rp_Runtime::N_THREADS_PER_TEAM; ++i) {
        // The order in which quantities are stored in the array must match the
        // order in which quantities are listed in the output file's header.
#ifdef DENS_VAR_C
        localIntQuantities_[0] += intQuantities_mass_[i];
        intQuantities_mass_[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C)
        localIntQuantities_[1] += intQuantities_xmom_[i];
        intQuantities_xmom_[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELY_VAR_C)
        localIntQuantities_[2] += intQuantities_ymom_[i];
        intQuantities_ymom_[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELZ_VAR_C)
        localIntQuantities_[3] += intQuantities_zmom_[i];
        intQuantities_zmom_[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(ENER_VAR_C)
        localIntQuantities_[4] += intQuantities_ener_[i];
        intQuantities_ener_[i] = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(VELX_VAR_C) && defined(VELY_VAR_C) && defined(VELZ_VAR_C)
        localIntQuantities_[5] += intQuantities_ke_[i];
        intQuantities_ke_[i]   = 0.0_wp;
#endif
#if defined(DENS_VAR_C) && defined(EINT_VAR_C)
        localIntQuantities_[6] += intQuantities_eint_[i];
        intQuantities_eint_[i] = 0.0_wp;
#endif
#ifdef MAGP_VAR_C
        localIntQuantities_[7] += intQuantities_magp_[i];
        intQuantities_magp_[i] = 0.0_wp;
#endif
    }

    if (MPI_Reduce((void*)localIntQuantities_,
                   (void*)globalIntQuantities_,
                   nIntQuantities_, MILHOJA_MPI_REAL, MPI_SUM,
                   ioRank_, MPI_COMM_WORLD) != MPI_SUCCESS) {
        throw std::runtime_error("[Io::reduceToIntegralQuantities] Unable to reduce integral quantities");
    }
}

/**
 * Write to file the integral quantities already computed by
 * reduceToGlobalIntegralQuantities.  The quantities are appended to the file given to
 * the IO unit at instantiation.
 *
 * \todo What are the requirements for writing integral quantities to file?
 *       Study the format strings to make certain that we are writing in a way
 *       similar to Fortran and such that there is no misprinted data up to the
 *       requirements.
 *
 * \param simTime - the simulation time at which the quantities were computed.
 */
void  Io::writeIntegralQuantities(const orchestration::Real simTime) {
    if (rank_ == ioRank_) {
        FILE*   fptr = fopen(intQuantitiesFile_.c_str(), "a");
        if (!fptr) {
            std::string msg =   "[Io::writeIntegralQuantities] ";
            msg += "Unable to open integral quantities output file ";
            msg += intQuantitiesFile_;
            throw std::runtime_error(msg);
        }

        fprintf(fptr, "%25.18e ", simTime);
        for (unsigned int i=0; i<nIntQuantities_; ++i) {
            fprintf(fptr, "%25.18e", globalIntQuantities_[i]);
            if (i < nIntQuantities_ - 1) {
                fprintf(fptr, " ");
            }
        }
        fprintf(fptr, "\n");

        fclose(fptr);
    }
}

