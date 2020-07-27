#include <iostream>

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>

#include "Grid.h"
#include "DataPacket.h"
#include "ThreadTeam.h"
#include "OrchestrationRuntime.h"

#include "Flash.h"
#include "constants.h"
#include "Analysis.h"
#include "setInitialConditions_block.h"
#include "scaleEnergy_block.h"
#include "computeLaplacianDensity_block.h"
#include "computeLaplacianEnergy_block.h"
#include "scaleEnergy_packet.h"
#include "computeLaplacianDensity_packet.h"
#include "computeLaplacianEnergy_packet.h"
#include "OrchestrationLogger.h"

#include "gtest/gtest.h"

using namespace orchestration;

namespace {

/**
 * Define a test fixture
 */
class TestRuntimePacket : public testing::Test {
protected:
    // Choose this so that it is not a divisor of the total number of blocks.
    // In this way, the last packet to be enqueued with a team will have fewer
    // blocks than the previous packets.
    static constexpr unsigned int   N_TILES_PER_PACKET = 500;

    RuntimeAction    computeLaplacianDensity_packet;
    RuntimeAction    computeLaplacianEnergy_packet;
    RuntimeAction    scaleEnergy_packet;
    RuntimeAction    computeErrors_packet;

    TestRuntimePacket(void) {
        computeLaplacianDensity_packet.name = "computeLaplacianDensity";
        computeLaplacianDensity_packet.nInitialThreads = 0;
        computeLaplacianDensity_packet.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianDensity_packet.nTilesPerPacket = 0;
        computeLaplacianDensity_packet.routine = ThreadRoutines::computeLaplacianDensity_packet;

        computeLaplacianEnergy_packet.name = "computeLaplacianEnergy";
        computeLaplacianEnergy_packet.nInitialThreads = 0;
        computeLaplacianEnergy_packet.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
        computeLaplacianEnergy_packet.nTilesPerPacket = 0;
        computeLaplacianEnergy_packet.routine = ThreadRoutines::computeLaplacianEnergy_packet;

        scaleEnergy_packet.name = "scaleEnergy";
        scaleEnergy_packet.nInitialThreads = 0;
        scaleEnergy_packet.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
        scaleEnergy_packet.nTilesPerPacket = 0;
        scaleEnergy_packet.routine = ThreadRoutines::scaleEnergy_packet;

        computeErrors_packet.name = "computeErrors";
        computeErrors_packet.nInitialThreads = 0;
        computeErrors_packet.teamType = ThreadTeamDataType::SET_OF_BLOCKS;
        computeErrors_packet.nTilesPerPacket = 0;
        computeErrors_packet.routine = Analysis::computeErrors_packet;

        Grid&    grid = Grid::instance();
        grid.initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                        N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                        NUNKVAR,
                        Simulation::setInitialConditions_block);
   }

    ~TestRuntimePacket(void) {
        Grid::instance().destroyDomain();
        // Let main routine finalize Grid unit/AMReX
    }
};

#ifndef DEBUG_RUNTIME
TEST_F(TestRuntimePacket, TestSinglePacketTeam) {
    amrex::MultiFab&   unk = Grid::instance().unk();

    orchestration::Logger::setLogFilename("TestSinglePacketTeam.log");

    constexpr unsigned int  N_THREADS = 4;
    ThreadTeam  cpu_packet(N_THREADS, 1);

    auto packet = std::shared_ptr<DataItem>{ new DataPacket{} };

    // Fix simulation to a single level and use AMReX 0-based indexing
    unsigned int   level = 0;

    try {
        computeLaplacianEnergy_packet.nInitialThreads = N_THREADS;
        computeLaplacianEnergy_packet.nTilesPerPacket = N_TILES_PER_PACKET;
        cpu_packet.startCycle(computeLaplacianEnergy_packet, "Cpu");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            packet->addSubItem( std::shared_ptr<DataItem>( new Tile{itor, level} ) );

            if (packet->nSubItems() >= computeLaplacianEnergy_packet.nTilesPerPacket) {
                cpu_packet.enqueue( std::move(packet) );
                packet = std::shared_ptr<DataItem>( new DataPacket{} );
            }
        }
        if (packet->nSubItems() != 0) {
            cpu_packet.enqueue( std::move(packet) );
            packet = std::shared_ptr<DataItem>( new DataPacket{} );
        }
        cpu_packet.closeQueue();
        cpu_packet.wait();

        computeLaplacianDensity_packet.nInitialThreads = N_THREADS;
        computeLaplacianDensity_packet.nTilesPerPacket = N_TILES_PER_PACKET - 2;
        cpu_packet.startCycle(computeLaplacianDensity_packet, "Cpu");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            packet->addSubItem( std::shared_ptr<DataItem>( new Tile{itor, level} ) );

            if (packet->nSubItems() >= computeLaplacianDensity_packet.nTilesPerPacket) {
                cpu_packet.enqueue( std::move(packet) );
                packet = std::shared_ptr<DataItem>( new DataPacket{} );
            }
        }
        if (packet->nSubItems() != 0) {
            cpu_packet.enqueue( std::move(packet) );
            packet = std::shared_ptr<DataItem>( new DataPacket{} );
        }
        cpu_packet.closeQueue();
        cpu_packet.wait();

        scaleEnergy_packet.nInitialThreads = N_THREADS;
        scaleEnergy_packet.nTilesPerPacket = N_TILES_PER_PACKET - 5;
        cpu_packet.startCycle(scaleEnergy_packet, "Cpu");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            packet->addSubItem( std::shared_ptr<DataItem>( new Tile{itor, level} ) );

            if (packet->nSubItems() >= scaleEnergy_packet.nTilesPerPacket) {
                cpu_packet.enqueue( std::move(packet) );
                packet = std::shared_ptr<DataItem>( new DataPacket{} );
            }
        }
        if (packet->nSubItems() != 0) {
            cpu_packet.enqueue( std::move(packet) );
            packet = std::shared_ptr<DataItem>( new DataPacket{} );
        }
        cpu_packet.closeQueue();
        cpu_packet.wait();

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        computeErrors_packet.nInitialThreads = N_THREADS;
        computeErrors_packet.nTilesPerPacket = N_TILES_PER_PACKET - 11;
        cpu_packet.startCycle(computeErrors_packet, "Cpu");
        for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
            packet->addSubItem( std::shared_ptr<DataItem>( new Tile{itor, level} ) );

            if (packet->nSubItems() >= computeErrors_packet.nTilesPerPacket) {
                cpu_packet.enqueue( std::move(packet) );
                packet = std::shared_ptr<DataItem>( new DataPacket{} );
            }
        }
        if (packet->nSubItems() != 0) {
            cpu_packet.enqueue( std::move(packet) );
        }
        cpu_packet.closeQueue();
        cpu_packet.wait();
        packet.reset();
    } catch (std::invalid_argument  e) {
        std::cerr << "\nINVALID ARGUMENT: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::logic_error  e) {
        std::cerr << "\nLOGIC ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::runtime_error  e) {
        std::cerr << "\nRUNTIME ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (...) {
        std::cerr << "\n??? ERROR: Unanticipated error\n\n";
        EXPECT_TRUE(false);
    }

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    Analysis::densityErrors(&L_inf1, &meanAbsErr1);
    Analysis::energyErrors(&L_inf2, &meanAbsErr2);

    EXPECT_TRUE(0.0 <= L_inf1);
    EXPECT_TRUE(L_inf1 <= 0.0);
    EXPECT_TRUE(0.0 <= meanAbsErr1);
    EXPECT_TRUE(meanAbsErr1 <= 0.0);

    EXPECT_TRUE(0.0 <= L_inf2);
    EXPECT_TRUE(L_inf2 <= 9.0e-6);
    EXPECT_TRUE(0.0 <= meanAbsErr2);
    EXPECT_TRUE(meanAbsErr2 <= 9.0e-6);
}
#endif

#ifndef DEBUG_RUNTIME
TEST_F(TestRuntimePacket, TestRuntimeSingle) {
    orchestration::Logger::setLogFilename("TestRuntimeSingle.log");

    ActionBundle    bundle;

    try {
        //***** FIRST RUNTIME EXECUTION CYCLE
        // Give an extra thread to the GPU task so that it can start to get work
        // to the postGpu task quicker.
        bundle.name                          = "Action Bundle 1";

        bundle.cpuAction.name                = "bundle1_cpuAction";
        bundle.cpuAction.nInitialThreads     = 1;
        bundle.cpuAction.teamType            = ThreadTeamDataType::BLOCK;
        bundle.cpuAction.nTilesPerPacket     = 0;
        bundle.cpuAction.routine             = ThreadRoutines::computeLaplacianDensity_block;

        bundle.gpuAction.name                = "bundle1_gpuAction";
        bundle.gpuAction.nInitialThreads     = 2;
        bundle.gpuAction.teamType            = ThreadTeamDataType::SET_OF_BLOCKS;
        bundle.gpuAction.nTilesPerPacket     = N_TILES_PER_PACKET;
        bundle.gpuAction.routine             = ThreadRoutines::computeLaplacianEnergy_packet;

        bundle.postGpuAction.name            = "bundle1_postGpuAction";
        bundle.postGpuAction.nInitialThreads = 1;
        bundle.postGpuAction.teamType        = ThreadTeamDataType::SET_OF_BLOCKS;
        bundle.postGpuAction.nTilesPerPacket = 0;
        bundle.postGpuAction.routine         = ThreadRoutines::scaleEnergy_packet;

        orchestration::Runtime::instance().executeTasks(bundle);

        //***** SECOND RUNTIME EXECUTION CYCLE
        bundle.name                          = "Analysis Bundle";

        bundle.cpuAction.name                = "";
        bundle.cpuAction.nInitialThreads     = 0;
        bundle.cpuAction.routine             = nullptr;

        bundle.gpuAction.name                = "computeErrors";
        bundle.gpuAction.nInitialThreads     = 4;
        bundle.gpuAction.teamType            = ThreadTeamDataType::SET_OF_BLOCKS;
        bundle.gpuAction.nTilesPerPacket     = N_TILES_PER_PACKET;
        bundle.gpuAction.routine             = Analysis::computeErrors_packet;

        bundle.postGpuAction.name            = "";
        bundle.postGpuAction.nInitialThreads = 0;
        bundle.postGpuAction.routine         = nullptr;

        Analysis::initialize(N_BLOCKS_X * N_BLOCKS_Y * N_BLOCKS_Z);
        orchestration::Runtime::instance().executeTasks(bundle);
    } catch (std::invalid_argument  e) {
        std::cerr << "\nINVALID ARGUMENT: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::logic_error  e) {
        std::cerr << "\nLOGIC ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (std::runtime_error  e) {
        std::cerr << "\nRUNTIME ERROR: "
                  << e.what() << "\n\n";
        EXPECT_TRUE(false);
    } catch (...) {
        std::cerr << "\n??? ERROR: Unanticipated error\n\n";
        EXPECT_TRUE(false);
    }

    double L_inf1      = 0.0;
    double meanAbsErr1 = 0.0;
    double L_inf2      = 0.0;
    double meanAbsErr2 = 0.0;
    Analysis::densityErrors(&L_inf1, &meanAbsErr1);
    Analysis::energyErrors(&L_inf2, &meanAbsErr2);
//    std::cout << "L_inf1 = " << L_inf1 << "\n";
//    std::cout << "L_inf2 = " << L_inf2 << std::endl;

    EXPECT_TRUE(0.0 <= L_inf1);
    EXPECT_TRUE(L_inf1 <= 1.0e-15);
    EXPECT_TRUE(0.0 <= meanAbsErr1);
    EXPECT_TRUE(meanAbsErr1 <= 1.0e-15);

    EXPECT_TRUE(0.0 <= L_inf2);
    EXPECT_TRUE(L_inf2 <= 9.0e-6);
    EXPECT_TRUE(0.0 <= meanAbsErr2);
    EXPECT_TRUE(meanAbsErr2 <= 9.0e-6);
}
#endif

}

