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
#include "ActionBundle.h"
#include "OrchestrationRuntime.h"

#include "Flash.h"
#include "constants.h"
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

void  setUp(void) {
    ActionBundle    bundle;
    bundle.name                          = "SetICs";
    bundle.cpuAction.name                = "initTile_cpu";
    bundle.cpuAction.nInitialThreads     = 4;
    bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
    bundle.cpuAction.routine             = Simulation::initTile_cpu;
    bundle.gpuAction.name                = "";
    bundle.gpuAction.nInitialThreads     = 0;
    bundle.gpuAction.teamType            = ThreadTeamDataType::BLOCK;
    bundle.gpuAction.routine             = nullptr;
    bundle.postGpuAction.name            = "";
    bundle.postGpuAction.nInitialThreads = 0;
    bundle.postGpuAction.teamType        = ThreadTeamDataType::BLOCK;
    bundle.postGpuAction.routine         = nullptr;

    OrchestrationRuntime::instance()->executeTasks(bundle);
}

void  tearDown(const std::string& filename,
               const std::string& mode,
               const unsigned int nLoops,
               const int nThdTask1, 
               const int nThdTask2, 
               const int nThdTask3,
               const double walltime) {
    int   rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        Grid*   grid = Grid::instance();
        amrex::MultiFab&  unk = grid->unk();
        amrex::Geometry   geometry = grid->geometry();
        amrex::Real       dx = geometry.CellSize(0);
        amrex::Real       dy = geometry.CellSize(1);
        grid = nullptr;

        double   L_inf_dens = 0.0;
        double   L_inf_ener = 0.0;
        double   meanAbsError = 0.0;
        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);

        ActionBundle    bundle;
        bundle.name                          = "ComputeErrors";
        bundle.cpuAction.name                = "computeErrors";
        bundle.cpuAction.nInitialThreads     = 3;
        bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.cpuAction.routine             = Analysis::computeErrors;
        bundle.gpuAction.name                = "";
        bundle.gpuAction.nInitialThreads     = 0;
        bundle.gpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.gpuAction.routine             = nullptr;
        bundle.postGpuAction.name            = "";
        bundle.postGpuAction.nInitialThreads = 0;
        bundle.postGpuAction.teamType        = ThreadTeamDataType::BLOCK;
        bundle.postGpuAction.routine         = nullptr;

        OrchestrationRuntime*   runtime = OrchestrationRuntime::instance();
        runtime->executeTasks(bundle);
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
}

int   main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "\nOne and only one command line argument please\n\n";
        return 1;
    }

    int  nTotalThreads = std::stoi(std::string(argv[1]));
    if (nTotalThreads <= 1) {
        std::cerr << "\nNeed to use at least two threads\n\n";
        return 2;
    }

    // Initialize simulation
    OrchestrationRuntime::setNumberThreadTeams(N_THREAD_TEAMS);
    OrchestrationRuntime::setMaxThreadsPerTeam(nTotalThreads);
    OrchestrationRuntime*   runtime = OrchestrationRuntime::instance();

    Grid*   grid = Grid::instance();
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
    fptr << "pmode,n_loops,n_thd_task1,n_thd_task2,n_thd_task3,"
         << "NXB,NYB,NZB,N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z,"
         << "dx,dy,Linf_density,Linf_energy,Walltime_sec\n";
    fptr.close();

    ActionBundle    bundle;
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
        bundle.name                          = "SerializedRuntimeBundle";
        bundle.cpuAction.name                = "computationalWork";
        bundle.cpuAction.nInitialThreads     = 1;
        bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.cpuAction.routine             = nullptr;
        bundle.gpuAction.name                = "";
        bundle.gpuAction.nInitialThreads     = 0;
        bundle.gpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.gpuAction.routine             = nullptr;
        bundle.postGpuAction.name            = "";
        bundle.postGpuAction.nInitialThreads = 0;
        bundle.postGpuAction.teamType        = ThreadTeamDataType::BLOCK;
        bundle.postGpuAction.routine         = nullptr;

        setUp();

        bundle.cpuAction.nInitialThreads = 1;
        bundle.cpuAction.routine         = ThreadRoutines::computeLaplacianDensity_cpu,
        tStart = MPI_Wtime(); 
        runtime->executeTasks(bundle);
        tWalltime = MPI_Wtime() - tStart; 

        bundle.cpuAction.nInitialThreads = 1;
        bundle.cpuAction.routine         = ThreadRoutines::computeLaplacianEnergy_cpu,
        tStart = MPI_Wtime(); 
        runtime->executeTasks(bundle);
        tWalltime += MPI_Wtime() - tStart; 

        bundle.cpuAction.nInitialThreads = 1;
        bundle.cpuAction.routine         = ThreadRoutines::scaleEnergy_cpu,
        tStart = MPI_Wtime(); 
        runtime->executeTasks(bundle);
        tWalltime += MPI_Wtime() - tStart; 

        tearDown(fname, "Runtime", 3, 1, 0, 0, tWalltime);

        /***** RUNTIME TEST - Try different thread combinations *****/
        bundle.name                          = "Action Bundle 1";
        bundle.cpuAction.name                = "bundle1_cpuAction";
        bundle.cpuAction.nInitialThreads     = 0;
        bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.cpuAction.routine             = ThreadRoutines::computeLaplacianDensity_cpu;
        bundle.gpuAction.name                = "bundle1_gpuAction";
        bundle.gpuAction.nInitialThreads     = 0;
        bundle.gpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.gpuAction.routine             = ThreadRoutines::computeLaplacianEnergy_cpu;
        bundle.postGpuAction.name            = "bundle1_postGpuAction";
        bundle.postGpuAction.nInitialThreads = 0;
        bundle.postGpuAction.teamType        = ThreadTeamDataType::BLOCK;
        bundle.postGpuAction.routine         = ThreadRoutines::scaleEnergy_cpu;

        for (unsigned int n=2; n<nTotalThreads; ++n) {
            int  nConcurrentThreads = n / 2;
            int  nPostThreads       = n % 2;

            setUp();
            bundle.cpuAction.nInitialThreads     = nConcurrentThreads;
            bundle.gpuAction.nInitialThreads     = nConcurrentThreads;
            bundle.postGpuAction.nInitialThreads = nPostThreads;

            tStart = MPI_Wtime(); 
            runtime->executeTasks(bundle);
            tWalltime = MPI_Wtime() - tStart; 

            tearDown(fname, "Runtime", 1,
                     nConcurrentThreads, nConcurrentThreads, nPostThreads,
                     tWalltime);
        }
    }

    grid->destroyDomain();
    delete grid;
    grid = nullptr;
    delete runtime;
    runtime = nullptr;

    return 0;
}

