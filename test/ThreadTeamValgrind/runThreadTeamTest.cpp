#include "ThreadTeam.h"

#include "DataItem.h"
#include "NullItem.h"

namespace TestThreadRoutines {
    void noop(const int tId, void* dataItem) {  };
}

int main(int argc, char* argv[]) {
    using namespace orchestration;

    constexpr unsigned int N_LOOPS = 100;
    constexpr unsigned int N_DATA  = 200;

    ThreadTeam  team1(2, 1);
    ThreadTeam  team2(4, 2);
    ThreadTeam  team3(6, 3);

    RuntimeAction    noop;
    noop.name = "noop";
    noop.teamType = ThreadTeamDataType::BLOCK;
    noop.routine = TestThreadRoutines::noop;

    std::shared_ptr<DataItem>   dataItem_1{};
    std::shared_ptr<DataItem>   dataItem_2{};

    for (unsigned int n=0; n<N_LOOPS; ++n) {
    	team1.attachThreadReceiver(&team3);
    	team2.attachThreadReceiver(&team3);
    	team2.attachDataReceiver(&team3);

    	noop.nInitialThreads = 2;
    	team1.startCycle(noop, "team1");

    	noop.nInitialThreads = 2;
    	team2.startCycle(noop, "team2");

    	noop.nInitialThreads = 0;
    	team3.startCycle(noop, "team3");

	for (int data=0; data<N_DATA; ++data) {
             dataItem_1 = std::shared_ptr<DataItem>( new NullItem{} );
             dataItem_2 = dataItem_1;

             team1.enqueue( std::move(dataItem_1) );
             team2.enqueue( std::move(dataItem_2) );
	}
    	team1.closeQueue();
    	team2.closeQueue();

    	team1.wait();
    	team2.wait();
    	team3.wait();

    	team1.detachThreadReceiver();
    	team2.detachThreadReceiver();
    	team2.detachDataReceiver();
    }

    return 0;
}

