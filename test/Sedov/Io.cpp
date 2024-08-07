#include "Io.h"

#include <stdexcept>

#include <mpi.h>

#include <Milhoja.h>
#include <Milhoja_Tile.h>
#include <Milhoja_Grid.h>
#include <Milhoja_Logger.h>

#include "Sedov.h"
#include "RuntimeParameters.h"

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
    milhoja::Logger::instance().log("[IO] Initializing...");

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

    milhoja::Logger::instance().log("[IO] Created and ready for use");
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
      localIntQuantities_{new milhoja::Real[nIntQuantities_]},
      globalIntQuantities_{nullptr},
      intQuantities_mass_{nullptr},
      intQuantities_xmom_{nullptr}, 
      intQuantities_ymom_{nullptr}, 
      intQuantities_zmom_{nullptr}, 
      intQuantities_ener_{nullptr}, 
      intQuantities_ke_{nullptr}, 
      intQuantities_eint_{nullptr}
{
    using namespace milhoja;

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
    RuntimeParameters&   RPs = RuntimeParameters::instance();
    unsigned int   nThreads{RPs.getUnsignedInt("Runtime", "nThreadsPerTeam")};
    intQuantities_mass_ = new Real[nThreads];
    intQuantities_xmom_ = new Real[nThreads];
    intQuantities_ymom_ = new Real[nThreads];
    intQuantities_zmom_ = new Real[nThreads];
    intQuantities_ener_ = new Real[nThreads];
    intQuantities_ke_   = new Real[nThreads];
    intQuantities_eint_ = new Real[nThreads];
    for (unsigned int i=0; i<nThreads; ++i) {
        intQuantities_mass_[i] = 0.0;
        intQuantities_xmom_[i] = 0.0;
        intQuantities_ymom_[i] = 0.0;
        intQuantities_zmom_[i] = 0.0;
        intQuantities_ener_[i] = 0.0;
        intQuantities_ke_[i]   = 0.0;
        intQuantities_eint_[i] = 0.0;
    }
    logger.log("[IO]     Mass");
    logger.log("[IO]     X-Momentum");
    logger.log("[IO]     Y-Momentum");
    logger.log("[IO]     Z-Momentum");
    logger.log("[IO]     Total Energy");
    logger.log("[IO]     Kinetic Energy");
    logger.log("[IO]     Internal Energy");

    // Append integral quantity header to file
    if (rank_ == ioRank_) {
        FILE*   fptr = fopen(intQuantitiesFile_.c_str(), "a");
        if (!fptr) {
            std::string msg = "[Io::Io] Unable to open integral quantities output file "
                              + intQuantitiesFile_;
            throw std::runtime_error(msg);
        }

        fprintf(fptr, "#%24s %25s %25s %25s %25s %25s %25s %25s\n",
                    "time", 
                    "mass",
                    "x-momentum",
                    "y-momentum",
                    "z-momentum",
                    "E_total",
                    "E_kinetic",
                    "E_internal");
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

    if (localIntQuantities_) {
        delete [] localIntQuantities_;
        localIntQuantities_ = nullptr;
    }

    if (globalIntQuantities_) {
        delete [] globalIntQuantities_;
        globalIntQuantities_ = nullptr;
    }

    intQuantitiesFile_ = "";
    comm_ = MPI_COMM_NULL;
    ioRank_ = -1;
}

void   Io::computeLocalIntegralQuantities(void) {
    using namespace milhoja;

    Grid&     grid = Grid::instance();

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
                                            const milhoja::IntVect& lo,
                                            const milhoja::IntVect& hi,
                                            const milhoja::FArray3D& cellVolumes,
                                            const milhoja::FArray4D& solnData) {
    //$milhoja  "solnData": {
    //$milhoja&    "R": [DENS_VAR, ENER_VAR, EINT_VAR,
    //$milhoja&          VELX_VAR, VELY_VAR, VELZ_VAR]
    //$milhoja& }

    milhoja::Real    dvol = 0.0;
    milhoja::Real    mass = 0.0;
    milhoja::Real    massSum = 0.0;
    milhoja::Real    xmomSum = 0.0;
    milhoja::Real    ymomSum = 0.0;
    milhoja::Real    zmomSum = 0.0;
    milhoja::Real    enerSum = 0.0;
    milhoja::Real    keSum   = 0.0;
    milhoja::Real    eintSum = 0.0;

    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            for (int i=lo.I(); i<=hi.I(); ++i) {
                dvol = cellVolumes(i, j, k);

                // mass
                mass = solnData(i, j, k, DENS_VAR) * dvol;
                massSum += mass;

                // momentum
                xmomSum += mass * solnData(i, j, k, VELX_VAR);
                ymomSum += mass * solnData(i, j, k, VELY_VAR);
                zmomSum += mass * solnData(i, j, k, VELZ_VAR);

                // total energy
                enerSum += mass * solnData(i, j, k, ENER_VAR);

                // kinetic energy
                keSum += 0.5_wp * mass
                                * (  std::pow(solnData(i, j, k, VELX_VAR), 2)
                                   + std::pow(solnData(i, j, k, VELY_VAR), 2)
                                   + std::pow(solnData(i, j, k, VELZ_VAR), 2));

              // internal energy
              eintSum += mass * solnData(i, j, k, EINT_VAR);
            }
        }
    }

    intQuantities_mass_[threadIdx] += massSum;
    intQuantities_xmom_[threadIdx] += xmomSum;
    intQuantities_ymom_[threadIdx] += ymomSum;
    intQuantities_zmom_[threadIdx] += zmomSum;
    intQuantities_ener_[threadIdx] += enerSum;
    intQuantities_ke_[threadIdx]   += keSum;
    intQuantities_eint_[threadIdx] += eintSum;
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
        localIntQuantities_[i] = 0.0;
    }

    // Any or all of the threads in the thread team that ran
    // computeIntegralQuantitiesByBlock could have computed part of the
    // integration.  Therefore, we integrate across all threads.
    RuntimeParameters&   RPs = RuntimeParameters::instance();
    unsigned int   nThreads{RPs.getUnsignedInt("Runtime", "nThreadsPerTeam")};
    for (unsigned int i=0; i<nThreads; ++i) {
        // The order in which quantities are stored in the array must match the
        // order in which quantities are listed in the output file's header.
        localIntQuantities_[0] += intQuantities_mass_[i];
        localIntQuantities_[1] += intQuantities_xmom_[i];
        localIntQuantities_[2] += intQuantities_ymom_[i];
        localIntQuantities_[3] += intQuantities_zmom_[i];
        localIntQuantities_[4] += intQuantities_ener_[i];
        localIntQuantities_[5] += intQuantities_ke_[i];
        localIntQuantities_[6] += intQuantities_eint_[i];
        intQuantities_mass_[i] = 0.0;
        intQuantities_xmom_[i] = 0.0;
        intQuantities_ymom_[i] = 0.0;
        intQuantities_zmom_[i] = 0.0;
        intQuantities_ener_[i] = 0.0;
        intQuantities_ke_[i]   = 0.0;
        intQuantities_eint_[i] = 0.0;
    }

    if (MPI_Reduce((void*)localIntQuantities_,
                   (void*)globalIntQuantities_,
                   nIntQuantities_, MILHOJA_MPI_REAL, MPI_SUM,
                   ioRank_, comm_) != MPI_SUCCESS) {
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
void  Io::writeIntegralQuantities(const milhoja::Real simTime) {
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

