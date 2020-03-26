#include <string>
#include <fstream>
#include <iostream>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Array4.H>

#include "Tile.h"
#include "Grid.h"
#include "OrchestrationRuntime.h"

#include "Flash.h"
#include "constants.h"
#include "estimateTimerResolution.h"
#include "initTile_cpu.h"
#include "Analysis.h"
#include "scaleEnergy_cpu.h"
#include "computeLaplacianDensity_cpu.h"
#include "computeLaplacianEnergy_cpu.h"

// Have make specify build-time constants for file headers
#include "buildInfo.h"

// No AMR.  Set level to 0-based AMReX coarse level
constexpr unsigned int   LEVEL = 0;

constexpr unsigned int   N_TRIALS = 5;
constexpr unsigned int   N_THREAD_TEAMS = 3;
constexpr unsigned int   N_THREADS_PER_TEAM = 4;

void  setUp(void) {
    OrchestrationRuntime::instance()->executeTasks("Task 1",
                                                   Simulation::initTile_cpu,
                                                   4, "cpuTask",
                                                   nullptr, 0, "null_gpuTask",
                                                   nullptr, 0, "null_postGpuTask");
}

void  tearDown(const std::string& filename,
               const std::string& mode,
               const unsigned int nLoops,
               const int nThdTask1, 
               const int nThdTask2, 
               const int nThdTask3,
               const double walltime) {
    Grid<NXB,NYB,NZB,NGUARD>*   grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
    amrex::MultiFab&  unk = grid->unk();
    amrex::Geometry   geometry = grid->geometry();
    amrex::Real       dx = geometry.CellSize(0);
    amrex::Real       dy = geometry.CellSize(1);
    grid = nullptr;

    double   L_inf_dens = 0.0;
    double   L_inf_ener = 0.0;
    double   meanAbsError = 0.0;
    Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);

    OrchestrationRuntime*   runtime = OrchestrationRuntime::instance();
    runtime->executeTasks("Task 1",
                          Analysis::computeErrors,
                          3, "cpuTask",
                          nullptr, 0, "null_gpuTask",
                          nullptr, 0, "null_postGpuTask");
    runtime = nullptr;

    Analysis::densityErrors(&L_inf_dens, &meanAbsError);
    Analysis::energyErrors(&L_inf_ener, &meanAbsError);

    std::ofstream  fptr;
    fptr.open(filename, std::ios::out | std::ios::app);
    fptr << std::setprecision(15) 
         << mode << ","
         << nLoops << ","
         << (nThdTask1 < 0 ? "NaN" : std::to_string(nThdTask1)) << ","
         << (nThdTask2 < 0 ? "NaN" : std::to_string(nThdTask2)) << ","
         << (nThdTask3 < 0 ? "NaN" : std::to_string(nThdTask3)) << ","
         << NXB << "," << NYB << "," << NZB << ","
         << N_BLOCKS_X << ","
         << N_BLOCKS_Y << ","
         << N_BLOCKS_Z << ","
         << dx << "," << dy << ","
         << L_inf_dens << "," << L_inf_ener << ","
         << walltime << std::endl;
    fptr.close();
}

int   main(int argc, char* argv[]) {
    // Initialize simulation
    OrchestrationRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    OrchestrationRuntime::setMaxThreadsPerTeam(N_THREADS_PER_TEAM);
    OrchestrationRuntime*   runtime = OrchestrationRuntime::instance();

    Grid<NXB,NYB,NZB,NGUARD>*   grid = Grid<NXB,NYB,NZB,NGUARD>::instance();
    grid->initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                     N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                     NUNKVAR,
                     Simulation::initTile_cpu);
    amrex::MultiFab&  unk = grid->unk();

    // Setup logging of results
    std::string  fname("gatherDataCpp_");
    fname += std::to_string(NXB) + "_";
    fname += std::to_string(NYB) + "_";
    fname += std::to_string(NZB) + "_";
    fname += std::to_string(N_BLOCKS_X) + "_";
    fname += std::to_string(N_BLOCKS_Y) + "_";
    fname += std::to_string(N_BLOCKS_Z) + ".dat";

    // Write header to file
    std::ofstream  fptr;
    fptr.open(fname, std::ios::out);
	fptr << "# Git repository," << PROJECT_GIT_REPO_NAME << std::endl;
    fptr << "# Git Commit," << PROJECT_GIT_REPO_VER << std::endl;
    fptr << "# AMReX version," << amrex::Version() << std::endl;
    fptr << "# C++ compiler," << CXX_COMPILER << std::endl;
    fptr << "# C++ compiler version," << CXX_COMPILER_VERSION << std::endl;
	fptr << "# Build date," << BUILD_DATETIME << std::endl;
	fptr << "# Hostname, " << HOSTNAME << std::endl;
    fptr << "# Host information," << MACHINE_INFO << std::endl;
    fptr << "# MPI_Wtick," << MPI_Wtick() << ",sec" << std::endl;
    fptr << "# Measured MPI_Wtime resolution," 
         << estimateTimerResolution() << ",sec"
         << std::endl;
    fptr << "pmode,n_loops,n_thd_task1,n_thd_task2,n_thd_task3,"
         << "NXB,NYB,NZB,N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z,"
         << "dx,dy,Linf_density,Linf_energy,Walltime_sec\n";
    fptr.close();

    for (unsigned int j=0; j<N_TRIALS; ++j) {
        /***** SERIAL TEST - Three iteration loops  *****/
        setUp();
        double tStart = MPI_Wtime(); 
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile     tileDesc(itor, LEVEL);
            ThreadRoutines::computeLaplacianDensity_cpu(0, &tileDesc);
        }

        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile     tileDesc(itor, LEVEL);
            ThreadRoutines::computeLaplacianEnergy_cpu(0, &tileDesc);
        }

        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile     tileDesc(itor, LEVEL);
            ThreadRoutines::scaleEnergy_cpu(0, &tileDesc);
        }
        double tWalltime = MPI_Wtime() - tStart;
        tearDown(fname, "Serial", 3, -1, -1, -1, tWalltime);

        /***** SERIAL TEST - Single iteration loop  *****/
        setUp();
        tStart = MPI_Wtime(); 
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            Tile     tileDesc(itor, LEVEL);
            ThreadRoutines::computeLaplacianDensity_cpu(0, &tileDesc);
            ThreadRoutines::computeLaplacianEnergy_cpu(0, &tileDesc);
            ThreadRoutines::scaleEnergy_cpu(0, &tileDesc);
        }
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "Serial", 1, -1, -1, -1, tWalltime);

        /***** RUNTIME TEST - Serialized version *****/
        setUp();
        tStart = MPI_Wtime(); 
        runtime->executeTasks("Task 1",
                              ThreadRoutines::computeLaplacianDensity_cpu,
                              1, "cpuTask",
                              nullptr, 0, "null_gpuTask",
                              nullptr, 0, "null_postGpuTask");
        runtime->executeTasks("Task 2",
                              ThreadRoutines::computeLaplacianEnergy_cpu,
                              1, "cpuTask",
                              nullptr, 0, "null_gpuTask",
                              nullptr, 0, "null_postGpuTask");
        runtime->executeTasks("Task 3",
                              ThreadRoutines::scaleEnergy_cpu,
                              1, "cpuTask",
                              nullptr, 0, "null_gpuTask",
                              nullptr, 0, "null_postGpuTask");
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "Runtime", 3, 1, 0, 0, tWalltime);

        /***** RUNTIME TEST - One thread per team *****/
        setUp();
        tStart = MPI_Wtime(); 
        runtime->executeTasks("Task Bundle 1",
                              ThreadRoutines::computeLaplacianDensity_cpu,
                              1, "bundle1_cpuTask",
                              ThreadRoutines::computeLaplacianEnergy_cpu,
                              1, "bundle1_gpuTask",
                              ThreadRoutines::scaleEnergy_cpu,
                              1, "bundle1_postGpuTask");
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "Runtime", 1, 1, 1, 1, tWalltime);

        /***** RUNTIME TEST - One thread per team *****/
        setUp();
        tStart = MPI_Wtime(); 
        runtime->executeTasks("Task Bundle 1",
                              ThreadRoutines::computeLaplacianDensity_cpu,
                              2, "bundle1_cpuTask",
                              ThreadRoutines::computeLaplacianEnergy_cpu,
                              2, "bundle1_gpuTask",
                              ThreadRoutines::scaleEnergy_cpu,
                              0, "bundle1_postGpuTask");
        tWalltime = MPI_Wtime() - tStart; 
        tearDown(fname, "Runtime", 1, 2, 2, 0, tWalltime);
    }

    grid->destroyDomain();
    delete grid;
    grid = nullptr;
    delete runtime;
    runtime = nullptr;

    return 0;
}

