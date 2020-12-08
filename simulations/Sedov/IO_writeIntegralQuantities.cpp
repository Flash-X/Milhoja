#include "IO.h"

#include <mpi.h>

#include "constants.h"

/**
 * Write the pre-computed integral quantities to file.  It is assumed that the
 * quantities of interest are stored in the IO unit's globalIntegralQuantities
 * array.  The quantities are appended to the file given to the IO unit at
 * initialization time.
 *
 * \todo Since simTime is in the Driver public interface, should we just use
 *       it internally rather than expect it as an argument?
 *
 * \param simTime - the simulation time at which the quantities were computed.
 */
void  IO::writeIntegralQuantities(const orchestration::Real  simTime) {
    int  rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == MASTER_PE) {
        FILE*   fptr = fopen(io::integralQuantitiesFilename.c_str(), "a");
        if (!fptr) {
            std::string msg =   "[IO::writeIntegralQuantities] ";
            msg += "Unable to open integral quantities output file ";
            msg += io::integralQuantitiesFilename;
            throw std::runtime_error(msg);
        }

        fprintf(fptr, "%25.18e ", simTime);
        for (unsigned int i=0; i<IO::nIntegralQuantities; ++i) {
            fprintf(fptr, "%25.18e", IO::globalIntegralQuantities[i]);
            if (i < IO::nIntegralQuantities - 1) {
                fprintf(fptr, " ");
            }
        }
        fprintf(fptr, "\n");

        fclose(fptr);
    }
}

