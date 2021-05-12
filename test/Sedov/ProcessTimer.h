#ifndef PROCESS_TIMER_H__
#define PROCESS_TIMER_H__

#include <string>
#include <fstream>

class ProcessTimer {
public:
    ProcessTimer(const std::string& filename,
                 const std::string& testname,
                 const unsigned int nDistributorThreads,
                 const unsigned int stagger_usec,
                 const unsigned int nCpuThreads,
                 const unsigned int nGpuThreads,
                 const unsigned int nBlocksPerPacket,
                 const unsigned int nBlocksPerCpuTurn);
    ~ProcessTimer(void);

    void logTimestep(const unsigned int step, const double wtime_sec);

private:
    int                 rank_;
    int                 nProcs_;
    std::ofstream       fptr_;
    double*             wtimes_sec_;
    unsigned int*       blockCounts_;
};

#endif

