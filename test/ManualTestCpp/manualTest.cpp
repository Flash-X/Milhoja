#include <thread>
#include <chrono>
#include <iostream>

#include "Tile.h"
#include "Grid.h"
#include "ThreadTeam.h"
#include "OrchestrationRuntime.h"

#include "Flash.h"
#include "constants.h"

using namespace orchestration;

void delay_1s(const int tId, int* work) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

void delay_1s(const int tId, Tile* work) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

/*
 * This test executes a short sequence of simple ThreadTeam and
 * OrchestrationRuntime actions with long pauses between each action.  The
 * intent is to manually confirm that resources (i.e. threads) are being
 * correctly created, used, and destroyed.
 *
 * Note that AMReX is explicitly used in this test so that one can
 * simultaneously get an idea of how it is using resources.
 */
int   main(int argc, char* argv[]) {
    //----- Start by just confirming one thread
    std::cout << "\nLead thread waiting for 5 seconds before doing anything ... ";
    std::flush(std::cout);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Let ThreadTeam create its threads
    std::cout << "Lead thread creates ThreadTeam<int> with 4 threads ... ";
    std::flush(std::cout);
    ThreadTeam<int>*  team = new ThreadTeam<int>(4, 1, "DeleteMe.log"); 
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";
  
    //----- Let ThreadTeam use some of its thread to do work
    std::cout << "ThreadTeam<int> executes 1s sleep task with 3 threads / 16 units of work ... ";
    std::flush(std::cout);
    int work = 1;
    team->startTask(delay_1s, 3, "teamName", "1s");
    for (unsigned int i=0; i<16; ++i) {
        work = i;
        team->enqueue(work, false);
    }
    team->closeTask();
    team->wait();
    std::cout << "done\n";

    //----- Delete thread team so that threads are destroyed
    std::cout << "Lead thread deletes ThreadTeam<int> with 4 threads ... ";
    std::flush(std::cout);
    delete team;
    team = nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Initailize AMReX so that we can see how many threads it uses
    std::cout << "Initialize AMReX ... ";
    std::flush(std::cout);
    Grid*    grid = Grid::instance();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Let AMReX do noop initialization of data
    std::cout << "Initialize Grid Domain ... ";
    std::flush(std::cout);
    grid->initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                     N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                     NUNKVAR,
                     delay_1s);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Create OrcehstrationRuntime
    std::cout << "Create OR with 3 TT of 4 threads each ... ";
    std::flush(std::cout);
    OrchestrationRuntime::setNumberThreadTeams(3);
    OrchestrationRuntime::setMaxThreadsPerTeam(4);
    OrchestrationRuntime*   runtime = OrchestrationRuntime::instance();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Let more than one TT do work
    std::cout << "Execute CPU/GPU concurrent 1s delay task ... ";
    std::flush(std::cout);
    runtime->executeTasks("Test bundle",
                          delay_1s, 3, "bundle1_cpuTask",
                          delay_1s, 3, "bundle1_gpuTask",
                          nullptr, 0, "null_postGpuTask");
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Release OR threads
    std::cout << "Destroy Orchestration Runtime ... ";
    std::flush(std::cout);
    delete runtime;
    runtime = nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Let AMReX finalize and clean-up
    std::cout << "Finalize AMReX and destroy Grid ... ";
    std::flush(std::cout);
    delete grid;
    grid = nullptr;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    //----- Pause so that we get final thread count
    std::cout << "Lead thread waiting for 5 seconds before terminating ... ";
    std::flush(std::cout);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "done\n";

    return 0;
}

