#include "ThreadTeam.h"

namespace TestThreadRoutines {
    void noop(const int tId, void* dataItem) {  };
}

int main(int argc, char* argv[]) {
    using namespace orchestration;

    constexpr unsigned int N_LOOPS = 100;
    constexpr unsigned int N_DATA  = 200;

    ThreadTeam<int>  team1(2, 1, "delete.me");
    ThreadTeam<int>  team2(4, 2, "delete.me");
    ThreadTeam<int>  team3(6, 3, "delete.me");

    RuntimeAction    noop;
    noop.name = "noop";
    noop.teamType = ThreadTeamDataType::BLOCK;
    noop.routine = TestThreadRoutines::noop;

    for (unsigned int n=0; n<N_LOOPS; ++n) {
    	team1.attachThreadReceiver(&team3);
    	team2.attachThreadReceiver(&team3);
    	team2.attachWorkReceiver(&team3);

    	noop.nInitialThreads = 2;
    	team1.startTask(noop, "team1");

    	noop.nInitialThreads = 2;
    	team2.startTask(noop, "team2");

    	noop.nInitialThreads = 0;
    	team3.startTask(noop, "team3");

	for (int data=0; data<N_DATA; ++data) {
             team1.enqueue(data, false);
             team2.enqueue(data, false);
	}
    	team1.closeTask();
    	team2.closeTask();

    	team1.wait();
    	team2.wait();
    	team3.wait();

    	team1.detachThreadReceiver();
    	team2.detachThreadReceiver();
    	team2.detachWorkReceiver();
    }

    return 0;
}

