#include <Milhoja_ThreadTeam.h>
#include <Milhoja_DataItem.h>
#include <Milhoja_DataPacket.h>
#include <DataItemSplitter.h>

#include "NullItem.h"

namespace TestThreadRoutines {
    void noop(const int tId, void* dataItem) {  };
}

int testThreadTeam(const unsigned int nLoops, const unsigned int nData) {
    using namespace milhoja;

    ThreadTeam  team1(2, 1);
    ThreadTeam  team2(4, 2);
    ThreadTeam  team3(6, 3);

    RuntimeAction    noop;
    noop.name = "noop";
    noop.teamType = ThreadTeamDataType::BLOCK;
    noop.routine = TestThreadRoutines::noop;

    std::shared_ptr<DataItem>   dataItem_1{};
    std::shared_ptr<DataItem>   dataItem_2{};

    for (unsigned int n=0; n<nLoops; ++n) {
    	team1.attachThreadReceiver(&team3);
    	team2.attachThreadReceiver(&team3);
    	team2.attachDataReceiver(&team3);

    	noop.nInitialThreads = 2;
    	team1.startCycle(noop, "team1");

    	noop.nInitialThreads = 2;
    	team2.startCycle(noop, "team2");

    	noop.nInitialThreads = 0;
    	team3.startCycle(noop, "team3");

	for (int data=0; data<nData; ++data) {
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

int testRuntimeFullPacket(const unsigned int nLoops,
		          const unsigned int nData,
			  const unsigned int nTilesPerPacket) {
    using namespace milhoja;

    ThreadTeam        team1{11, 1};
    ThreadTeam        team2{20, 2};
    ThreadTeam        team3{32, 3};
    DataItemSplitter  splitter{};

    // Specify action bundle including map of action routine
    // to memory system in which data is expected to reside
    RuntimeAction    p1_actionA_block_cpu;
    RuntimeAction    p2_actionB_DPblock_gpu;
    RuntimeAction    p2_actionC_block_cpu;

    p1_actionA_block_cpu.name = "Action A";
    p1_actionA_block_cpu.nInitialThreads = 11;
    p1_actionA_block_cpu.teamType = ThreadTeamDataType::BLOCK;
    p1_actionA_block_cpu.nTilesPerPacket = 0;
    p1_actionA_block_cpu.routine = TestThreadRoutines::noop;

    p2_actionB_DPblock_gpu.name = "Action B";
    p2_actionB_DPblock_gpu.nInitialThreads = 20;
    p2_actionB_DPblock_gpu.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
    p2_actionB_DPblock_gpu.nTilesPerPacket = nTilesPerPacket;
    p2_actionB_DPblock_gpu.routine = TestThreadRoutines::noop;

    p2_actionC_block_cpu.name = "Action C";
    p2_actionC_block_cpu.nInitialThreads = 1;
    p2_actionC_block_cpu.teamType = ThreadTeamDataType::BLOCK;
    p2_actionC_block_cpu.nTilesPerPacket = 0;
    p2_actionC_block_cpu.routine = TestThreadRoutines::noop;

    for (unsigned int n=0; n<nLoops; ++n) {
    	// Setup desired thread team configuration with data item splitter
    	team1.attachThreadReceiver(&team3);	
    	team2.attachThreadReceiver(&team3);	
    	team2.attachDataReceiver(&splitter);
    	splitter.attachDataReceiver(&team3);
	
	// Map action bundle onto thread team configuration
    	team1.startCycle(p1_actionA_block_cpu,   "cpuTeam");
    	team2.startCycle(p2_actionB_DPblock_gpu, "gpuTeam");
    	team3.startCycle(p2_actionC_block_cpu,   "postGpuTeam");

    	// Start giving data items to the configuration
    	std::shared_ptr<DataItem>   dataItem_cpu{};
    	std::shared_ptr<DataItem>   dataItem_gpu{ new DataPacket{} };
    	for (int data=0; data<nData; ++data) {
    	     dataItem_cpu = std::shared_ptr<DataItem>( new NullItem{} );
    	     dataItem_gpu->addSubItem( std::shared_ptr<DataItem>{ dataItem_cpu } );

    	     team1.enqueue( std::move(dataItem_cpu) );
    	     if (dataItem_gpu->nSubItems() >= nTilesPerPacket) {
    	         team2.enqueue( std::move(dataItem_gpu) );
    	         dataItem_gpu = std::shared_ptr<DataItem>{ new DataPacket{} };
    	     }
    	}
    	team1.closeQueue();

    	if (dataItem_gpu->nSubItems() > 0) {
    	    team2.enqueue( std::move(dataItem_gpu) );
    	} else {
    	    dataItem_gpu.reset();
    	}
    	team2.closeQueue();

    	team1.wait();
    	team2.wait();
    	team3.wait();

    	// Break up configuration
    	team1.detachThreadReceiver();
    	team2.detachThreadReceiver();
    	team2.detachDataReceiver();
    	splitter.detachDataReceiver();
    }

    return 0;
}

int main(int argc, char* argv[]) {
    constexpr unsigned int N_TILES_PER_PACKET = 50;
    constexpr unsigned int N_LOOPS = 100;
    constexpr unsigned int N_DATA  = 200;

    int errCode = testThreadTeam(N_LOOPS, N_DATA);
    errCode     = testRuntimeFullPacket(N_LOOPS,
		                        N_DATA,
					N_TILES_PER_PACKET);

    // TODO: Run a test for each thread team configuration in the runtime.

    return 0;
}

