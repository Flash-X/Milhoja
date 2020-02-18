#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <pthread.h>

#include "testThreadRoutines.h"
#include "OrchestrationRuntime.h"

#include "gtest/gtest.h"

namespace {
/*
 * Test if a ThreadTeam object is destroyed correctly even if the mode isn't in
 * Idle (i.e. there are at least some threads in the Wait state).
 *
 * As presently implemented, this cannot be tested automatically.  Rather, we
 * should not see an error reports in the test output and we need to manually
 * study the log file to confirm that the destruction happened as expected.
 *
 * \todo - figure out how to test this automatically
 */
TEST(RuntimeTest, TestAbnormalDestroy) {
    // Test in Running & Open 
    ThreadTeam*    team1 = new ThreadTeam(4, 1, "TestAbnormalDestroy.log");

    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1->mode());
        
    team1->startTask(TestThreadRoutines::delay_500ms, 3, "teamName", "500ms");
    team1->enqueue(1);

    // Give team time to activate the threads, but for destruction to 
    // occur while we have a thread computing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1->mode());

    // Set to terminate
    delete  team1;
    team1 = nullptr;
 
    // Test in Running & Closed or Running w/ NoMoreWork
    team1 = new ThreadTeam(4, 1, "TestAbnormalDestroy.log");

    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1->mode());
        
    team1->startTask(TestThreadRoutines::delay_500ms, 3, "teamName", "500ms");
    team1->enqueue(1);
    team1->closeTask();

    // Give team time to activate the threads, but for destruction to 
    // occur while we have a thread computing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_NE(ThreadTeam::MODE_IDLE, team1->mode());

    // Set to terminate
    delete  team1;
    team1 = nullptr;
}

TEST(RuntimeTest, TestIdleNoRun) {
    unsigned int   N_ITERS = 10;

    for (unsigned int i=0; i<N_ITERS; ++i) {
        ThreadTeam    team1(10, 1, "TestIdleNoRun.log");
        ThreadTeam    team2(5,  2, "TestIdleNoRun.log");
        ThreadTeam    team3(2,  3, "TestIdleNoRun.log");
   
        EXPECT_EQ(10, team1.nMaximumThreads());
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
        
        EXPECT_EQ(5,  team2.nMaximumThreads());
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team2.mode());
        
        EXPECT_EQ(2,  team3.nMaximumThreads());
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team3.mode());

        // This should not block and therefore repeated calls should pass
        team1.wait();
        team1.wait();

        // Ask for more threads than in team
        ASSERT_THROW(team1.startTask(nullptr, 11, "teamName", "null"),
                     std::logic_error);
        ASSERT_THROW(team1.startTask(TestThreadRoutines::noop, 11, "teamName", "noop"),
                     std::logic_error);
        ASSERT_THROW(team1.increaseThreadCount(0),  std::logic_error);
        ASSERT_THROW(team1.increaseThreadCount(11), std::logic_error);

        // Use methods that are not allowed in Idle
        ASSERT_THROW(team1.enqueue(1),  std::runtime_error);
        ASSERT_THROW(team1.closeTask(), std::runtime_error);

        // Detach when no teams have been attached
        ASSERT_THROW(team1.detachThreadReceiver(), std::logic_error);
        ASSERT_THROW(team1.detachWorkReceiver(),   std::logic_error);

        // Attach null team
        ASSERT_THROW(team1.attachThreadReceiver(nullptr), std::logic_error);
        ASSERT_THROW(team1.attachWorkReceiver(nullptr),   std::logic_error);

        // Attach team to itself
        ASSERT_THROW(team1.attachThreadReceiver(&team1), std::logic_error);
        ASSERT_THROW(team1.attachWorkReceiver(&team1),   std::logic_error);

        // Setup basic topology
        team1.attachThreadReceiver(&team3);
        team2.attachThreadReceiver(&team3);
        team2.attachWorkReceiver(&team3);

        // Not allowed to attach more than one receiver
        ASSERT_THROW(team1.attachThreadReceiver(&team3), std::logic_error);
        ASSERT_THROW(team2.attachThreadReceiver(&team3), std::logic_error);
        ASSERT_THROW(team2.attachWorkReceiver(&team3),   std::logic_error);

        // Break down topology so that destruction is clean
        team1.detachThreadReceiver();
        team2.detachThreadReceiver();
        team2.detachWorkReceiver();

        // If these were properly detached above, these should fail
        ASSERT_THROW(team1.detachThreadReceiver(), std::logic_error);
        ASSERT_THROW(team2.detachThreadReceiver(), std::logic_error);
        ASSERT_THROW(team2.detachWorkReceiver(),   std::logic_error);
    }
}

TEST(RuntimeTest, TestRunningOpenNoWork) {
    unsigned int  N_ITERS = 100;

    ThreadTeam  team1(10, 1, "TestRunningOpenNoWork.log");
    ThreadTeam  team2(2,  2, "TestRunningOpenNoWork.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        EXPECT_EQ(10, team1.nMaximumThreads());
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());

        team1.startTask(TestThreadRoutines::noop, 5, "test", "noop");
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());

        // Call methods that are not allowed in Running & Open
        ASSERT_THROW(team1.startTask(TestThreadRoutines::noop, 5, "test", "noop"),
                     std::runtime_error);
        ASSERT_THROW(team1.attachThreadReceiver(&team2), std::logic_error);
        ASSERT_THROW(team1.detachThreadReceiver(),       std::logic_error);
        ASSERT_THROW(team1.attachWorkReceiver(&team2),   std::logic_error);
        ASSERT_THROW(team1.detachWorkReceiver(),         std::logic_error);

        ASSERT_THROW(team1.increaseThreadCount(0),  std::logic_error);
        ASSERT_THROW(team1.increaseThreadCount(11), std::logic_error);

        team1.closeTask();
        team1.wait();
    }
}

TEST(RuntimeTest, TestNoWork) {
    unsigned int  N_ITERS = 100;

    ThreadTeam  team1(10, 1, "TestNoWork.log");

    for (unsigned int i=0; i<N_ITERS; ++i) {
        EXPECT_EQ(10, team1.nMaximumThreads());
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());

        team1.startTask(TestThreadRoutines::noop, 5, "test", "noop");
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());

        team1.closeTask();
        team1.wait();
        EXPECT_EQ(10, team1.nMaximumThreads());
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    }
}
}

