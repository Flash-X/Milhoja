#include "ProcessTimer.h"

#include <iomanip>

#include <milhoja.h>
#include <Milhoja_Grid.h>

#include "Flash_par.h"

/**
 *
 */
ProcessTimer::ProcessTimer(const std::string& filename,
                           const std::string& testname,
                           const unsigned int nDistributorThreads,
                           const unsigned int stagger_usec,
                           const unsigned int nCpuThreads,
                           const unsigned int nGpuThreads,
                           const unsigned int nBlocksPerPacket,
                           const unsigned int nBlocksPerCpuTurn,
                           const MPI_Comm comm,
                           const int timerRank)
    : comm_{comm},
      timerRank_{timerRank},
      rank_{-1},
      nProcs_{-1},
      fptr_{},
      wtimes_sec_{nullptr},
      blockCounts_{nullptr}
{
    if (timerRank_ < 0) {
        throw std::invalid_argument("[ProcessTimer::ProcessTimer] Negative MPI rank");
    }

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nProcs_);

    unsigned int    nxb, nyb, nzb;
    milhoja::Grid::instance().getBlockSize(&nxb, &nyb, &nzb);

    // Write header to file
    if (rank_ == timerRank_) {
        wtimes_sec_  = new double[nProcs_];
        blockCounts_ = new unsigned int[nProcs_];

        fptr_.open(filename, std::ios::out);
        fptr_ << "# Testname = " << testname << "\n";
        fptr_ << "# Dimension = " << NDIM << "\n";
        fptr_ << "# NXB = " << nxb << "\n";
        fptr_ << "# NYB = " << nyb << "\n";
        fptr_ << "# NZB = " << nzb << "\n";
        fptr_ << "# N_BLOCKS_X = " << rp_Grid::N_BLOCKS_X << "\n";
        fptr_ << "# N_BLOCKS_Y = " << rp_Grid::N_BLOCKS_Y << "\n";
        fptr_ << "# N_BLOCKS_Z = " << rp_Grid::N_BLOCKS_Z << "\n";
        fptr_ << "# n_distributor_threads = " << nDistributorThreads << "\n";
        fptr_ << "# stagger_usec = " << stagger_usec << "\n";
        fptr_ << "# n_cpu_threads = " << nCpuThreads << "\n";
        fptr_ << "# n_gpu_threads = " << nGpuThreads << "\n";
        fptr_ << "# n_blocks_per_packet = " << nBlocksPerPacket << "\n";
        fptr_ << "# n_blocks_per_cpu_turn = " << nBlocksPerCpuTurn << "\n";
        fptr_ << "# MPI_Wtick_sec = " << MPI_Wtick() << "\n";
        fptr_ << "# step,nblocks_1,walltime_sec_1,...,nblocks_N,walltime_sec_N\n";
    }
}

/**
 *
 */
ProcessTimer::~ProcessTimer(void) {
    if (rank_ == timerRank_) {
        fptr_.close();
    }

    if (wtimes_sec_) {
        delete [] wtimes_sec_;
        wtimes_sec_ = nullptr;
    }

    if (blockCounts_) {
        delete [] blockCounts_;
        blockCounts_ = nullptr;
    }
}

/**
 *
 */
void ProcessTimer::logTimestep(const unsigned int step, const double wtime_sec) {
    unsigned int nBlocks = milhoja::Grid::instance().getNumberLocalBlocks();
    MPI_Gather(&wtime_sec,  1, MPI_DOUBLE,
               wtimes_sec_, 1, MPI_DOUBLE, timerRank_,
               comm_);
    MPI_Gather(&nBlocks,     1, MPI_UNSIGNED,
               blockCounts_, 1, MPI_UNSIGNED, timerRank_,
               comm_);

    if (rank_ == timerRank_) {
        fptr_ << std::setprecision(15) 
              << step << ",";
        for (unsigned int proc=0; proc<nProcs_; ++proc) {
            fptr_ << blockCounts_[proc] << ',' << wtimes_sec_[proc];
            if (proc < nProcs_ - 1) {
                fptr_ << ',';
            }
        }
    }
    fptr_ << std::endl;
}

