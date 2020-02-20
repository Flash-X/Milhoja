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

/**
 *  Test that teams are created and start in the appropriate initial state.
 *  This implicitly tests destruction from the Idle mode.
 */
TEST(ThreadTeamTest, TestInitialState) {
    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam    team1(3, 1, "TestInitialState.log");
    ThreadTeam    team2(2, 2, "TestInitialState.log");

    // Confirm explicit state
    team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(3, team1.nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    EXPECT_EQ(3, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(0, N_Q);

    team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(2, team2.nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team2.mode());
    EXPECT_EQ(2, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(0, N_Q);

    // Confirm indirectly that no subscribers are attached
    ASSERT_THROW(team1.detachThreadReceiver(), std::logic_error);
    ASSERT_THROW(team1.detachWorkReceiver(),   std::logic_error);

    ASSERT_THROW(team2.detachThreadReceiver(), std::logic_error);
    ASSERT_THROW(team2.detachWorkReceiver(),   std::logic_error);
}

/**
 * Test if a ThreadTeam object is destroyed correctly even if the mode isn't in
 * Idle (i.e. there are at least some threads in the Wait state).
 *
 * As presently implemented, this cannot be tested automatically.  Rather, we
 * should not see an error reports in the test output and we need to manually
 * study the log file to confirm that the destruction happened as expected.
 *
 * \todo - figure out how to test this automatically
 */
TEST(ThreadTeamTest, TestDestruction) {
    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    // Test in Running & Open 
    ThreadTeam*    team1 = new ThreadTeam(4, 1, "TestDestruction.log");
    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1->mode());

    team1->startTask(TestThreadRoutines::delay_100ms, 3, "teamName", "100ms");
    team1->enqueue(1);
    for (unsigned int i=0; i<100; ++i) {
        team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_comp == 1)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1->mode());
    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(1, N_idle);
    EXPECT_EQ(2, N_wait);
    EXPECT_EQ(1, N_comp);
    EXPECT_EQ(0, N_Q);

    // Set to terminate
    delete  team1;
    team1 = nullptr;
 
    // Test in Running & Closed or Running w/ NoMoreWork
    team1 = new ThreadTeam(4, 1, "TestDestruction.log");
    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1->mode());
 
    // Wait for a thread to dequeue the work before closing queue
    // The mode will transition to Running & No More Work as there
    // won't be any work in the queue
    team1->startTask(TestThreadRoutines::delay_100ms, 3, "teamName", "100ms");
    team1->enqueue(1);
    for (unsigned int i=0; i<100; ++i) {
        team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_Q == 0)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Close task & let Waiting threads go Idle
    team1->closeTask();
    for (unsigned int i=0; i<100; ++i) {
        team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_idle == 3)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team1->mode());
    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(3, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(1, N_comp);
    EXPECT_EQ(0, N_Q);

    // Set to terminate
    delete  team1;
    team1 = nullptr;
}

/**
 * Calling wait() in the Idle state should be a no-op.  While necessary (see
 * documentation), it does allow for nonsensical usage.  We confirm that
 * nonsense is allowed.
 */
TEST(ThreadTeamTest, TestIdleWait) {
    ThreadTeam    team1(3, 1, "TestIdleWait.log");
    
    // Call wait without having run a task
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    team1.wait();
    team1.wait();
    team1.wait();

    // Do an execution cycle with no threads/no work and call wait 
    // as many times as we want
    team1.startTask(TestThreadRoutines::noop, 0, "test1", "noop");
    team1.closeTask();
    for (unsigned int i=0; i<100; ++i) {
        if (team1.mode() == ThreadTeam::MODE_IDLE)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    team1.wait();
    team1.wait();
    team1.wait();

    // Do an execution cycle with work
    team1.startTask(TestThreadRoutines::noop, 1, "test1", "noop");
    team1.enqueue(1);
    team1.closeTask();
    for (unsigned int i=0; i<100; ++i) {
        if (team1.mode() == ThreadTeam::MODE_IDLE)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    team1.wait();
    team1.wait();
    team1.wait();
}

/**
 *  Create a team.  Start a task without threads and close the task without adding
 *  work.  While this is odd, it should be acceptable.
 *
 *  Configure this team with a work subscriber so that we confirm that it
 *  transitions to Idle as well automatically.
 */
TEST(ThreadTeamTest, TestNoWorkNoThreads) {
    ThreadTeam    team1(3, 1, "TestNoWorkNoThreads.log");
    ThreadTeam    team2(2, 2, "TestNoWorkNoThreads.log");

    EXPECT_EQ(3, team1.nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());

    EXPECT_EQ(2, team2.nMaximumThreads());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team2.mode());

    team1.attachWorkReceiver(&team2);

    team1.startTask(TestThreadRoutines::noop, 0, "test1", "noop");
    team2.startTask(TestThreadRoutines::noop, 0, "test2", "noop");
    team1.closeTask();
    team1.wait();
    // Next call will hang if team1 doesn't all team2's closeTask()
    team2.wait();

    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team2.mode());
}

/**
 * Confirm that the thread team responds to incorrect use of team's interface in
 * the Idle mode.
 */
TEST(ThreadTeamTest, TestIdleErrors) {
    unsigned int   N_ITERS = 10;

    // Confirm correct construction & initial state
    ThreadTeam    team1(10, 1, "TestIdleErrors.log");
    ThreadTeam    team2(5,  2, "TestIdleErrors.log");
    ThreadTeam    team3(2,  3, "TestIdleErrors.log");

    // Repeat several times so we confirm in the case that we have and haven't
    // run execution cycles
    for (unsigned int i=0; i<N_ITERS; ++i) {
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

        team1.startTask(TestThreadRoutines::noop, 5, "test1", "noop");
        team1.enqueue(1);
        team1.closeTask();
        team1.wait();
    }
}

/**
 *  Confirm that threads are forwarded properly by a team that is in Idle
 *  and that a team in Running & Closed only uses those threads it receives and
 *  forwards along the rest.
 */ 
TEST(ThreadTeamTest, TestIdleForwardsThreads) {
    unsigned int  N_ITERS = 100;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(2, 1, "TestIdleForwardsThreads.log");
    ThreadTeam  team2(4, 2, "TestIdleForwardsThreads.log");
    ThreadTeam  team3(6, 3, "TestIdleForwardsThreads.log");

    team1.attachThreadReceiver(&team2);
    team2.attachThreadReceiver(&team3);

    // Team 1 stays in Idle
    // Team 2 setup in Running & Closed with one pending item
    //        It will take Team 2 some time to finish its 
    //        work once it gets a thread
    // Team 3 setup in Running & Closed with one pending item
    //        It will finish quickly once it gets a thread
    team2.startTask(TestThreadRoutines::delay_100ms, 0, "wait", "100ms");
    team3.startTask(TestThreadRoutines::noop, 0, "quick", "noop");
    team2.enqueue(1);
    team3.enqueue(2);
    team2.closeTask();
    team3.closeTask();

    team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    EXPECT_EQ(2, team1.nMaximumThreads());
    EXPECT_EQ(2, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(0, N_Q);

    team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team2.mode());
    EXPECT_EQ(4, team2.nMaximumThreads());
    EXPECT_EQ(4, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(1, N_Q);

    team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team3.mode());
    EXPECT_EQ(6, team3.nMaximumThreads());
    EXPECT_EQ(6, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(1, N_Q);

    // Team 1 should just forward 2 threads to Team 2
    // Team 2 should determine that it just needs 1 and foward other to Team 2
    //   => Team 3 should finish before Team 2
    team1.increaseThreadCount(2);

    // Let team 3 finish and give Team 2 time to start work
    team3.wait();
    for (unsigned int i=0; i<10; ++i) {
        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_comp == 1)    break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team2.mode());
    EXPECT_EQ(4, team2.nMaximumThreads());
    EXPECT_EQ(3, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(1, N_comp);
    EXPECT_EQ(0, N_Q);

    team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team3.mode());
    EXPECT_EQ(6, team3.nMaximumThreads());
    EXPECT_EQ(6, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(0, N_Q);

    team2.wait();

    EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team2.mode());
    EXPECT_EQ(ThreadTeam::MODE_IDLE, team3.mode());
}

/**
 * Confirm proper functionality when no work is given, but we still run cycles.
 */
TEST(ThreadTeamTest, TestNoWork) {
    unsigned int  N_ITERS = 100;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(5, 1, "TestNoWork.log");

    for (unsigned int i=0; i<N_ITERS; ++i) {
        team1.startTask(TestThreadRoutines::noop, 3, "test", "noop");
   
        for (unsigned int i=0; i<100; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 3)  break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(5, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(3, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team1.closeTask();
        team1.wait();

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
        EXPECT_EQ(5, team1.nMaximumThreads());
        EXPECT_EQ(5, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);
    }
}

TEST(ThreadTeamTest, TestRunningOpenErrors) {
    unsigned int  N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(10, 1, "TestRunningOpenNoWork.log");
    ThreadTeam  team2(2,  2, "TestRunningOpenNoWork.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        team1.startTask(TestThreadRoutines::noop, 5, "test", "noop");
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());

        // Wait some time for Idle threads to activate
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 5)     break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(10, team1.nMaximumThreads());
        EXPECT_EQ(5, N_idle);
        EXPECT_EQ(5, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

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

/**
 *  Confirm that threads can be increased correctly in Running & Open.
 */
TEST(ThreadTeamTest, TestRunningOpenIncreaseThreads) {
    unsigned int  N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(3, 1, "TestRunningOpenIncreaseThreads.log");
    ThreadTeam  team2(4, 2, "TestRunningOpenIncreaseThreads.log");

    team1.attachThreadReceiver(&team2);

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        // All teams should generally have startTask called before we start
        // interacting with any team (e.g. increasing threads or adding work)
        team1.startTask(TestThreadRoutines::noop, 0, "test1", "noop");
        team2.startTask(TestThreadRoutines::noop, 0, "test2", "noop");
        team1.enqueue(1);

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(1, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(4, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Increasing count in Open means that Team 1 should keep all that we
        // have given even though we have more waiting threads than work in the
        // queue
        team1.increaseThreadCount(2);

        for (unsigned int i=0; i<100; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 0)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(4, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // This should result in threads being sent to Team 2
        team1.closeTask();
        team1.wait();

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_IDLE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.closeTask();
        team2.wait();
    }

    team1.detachThreadReceiver();
}

/**
 *  Confirm that calling enqueue with all non-Idle threads in Wait results in a
 *  Wait thread transitioning to a Compute thread.  Also make certain that
 *  computing threads push work to Work subscribers.
 */
TEST(ThreadTeamTest, TestRunningOpenEnqueue) {
    unsigned int  N_ITERS = 2;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(3, 1, "TestRunningOpenEnqueue.log");
    ThreadTeam  team2(2, 2, "TestRunningOpenEnqueue.log");

    team1.attachWorkReceiver(&team2);

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        team1.startTask(TestThreadRoutines::delay_100ms, 1, "wait",  "100ms");
        team2.startTask(TestThreadRoutines::delay_100ms, 2, "wait",  "100ms");
        for (unsigned int i=0; i<100; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 3)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(1, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Enqueue two units of work so that the single thread will necessarily
        // finish work on one, emit/receive computationFinished and find the
        // next unit of work
        team1.enqueue(1);
        team1.enqueue(2);
        for (unsigned int i=0; i<100; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(1, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(2, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        for (unsigned int i=0; i<1000; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 0)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (unsigned int i=0; i<100; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        // Confirm that team 2 got a unit of work
        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(2, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(1, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        // Thread team 1 will call closeTask for team 2
        team1.closeTask();
        team1.wait();
        team2.wait();
    }

    team1.detachWorkReceiver();
}

/**
 * Confirm that ThreadTeam interface methods that should fail in the Running &
 * Closed mode do fail.
 */
TEST(ThreadTeamTest, TestRunningClosedErrors) {
    unsigned int  N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(5, 1, "TestRunningClosedErrors.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        // Team must have one threads and two units of work to stay to closed
        team1.startTask(TestThreadRoutines::delay_100ms, 1, "wait",  "100ms");
        team1.enqueue(1);
        team1.enqueue(2);
        team1.closeTask();

        EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team1.mode());
        ASSERT_THROW(team1.startTask(TestThreadRoutines::noop, 0,
                                     "quick",  "fail"), std::runtime_error);
        ASSERT_THROW(team1.startTask(TestThreadRoutines::noop, 1,
                                     "quick",  "fail"), std::runtime_error);

        ASSERT_THROW(team1.increaseThreadCount(0), std::logic_error);
        ASSERT_THROW(team1.increaseThreadCount(team1.nMaximumThreads()+1),
                                               std::logic_error);

        ASSERT_THROW(team1.enqueue(1),  std::runtime_error);
        ASSERT_THROW(team1.closeTask(), std::runtime_error);

        // Make certain that all of the above was still done in Running & Closed
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team1.mode());

        team1.wait();
    }
}

/**
 * Confirm that activating threads when in Running & Closed works as expected.
 */
TEST(ThreadTeamTest, TestRunningClosedActivation) {
    unsigned int  N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;
    
    ThreadTeam  team1(3, 1, "TestRunningClosedActivation.log");

    for (unsigned int i=0; i<N_ITERS; ++i) {
        team1.startTask(TestThreadRoutines::delay_100ms, 1, "wait",  "100ms");
        team1.enqueue(1);
        team1.enqueue(2);
        team1.enqueue(3);
        team1.closeTask();
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 2)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(2, N_Q);

        // Activate a thread so that it can start computing but not transition the
        // mode
        team1.increaseThreadCount(1);
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(2, N_comp);
        EXPECT_EQ(1, N_Q);

        // Activate final thread so that it can start computing and transition mode
        team1.increaseThreadCount(1);
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 0)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(3, N_comp);
        EXPECT_EQ(0, N_Q);

        team1.wait();
    }
}

/**
 * Confirm that activating threads when in Running & Closed works as expected.
 */
TEST(ThreadTeamTest, TestRunningClosedWorkPub) {
    unsigned int  N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;
    
    ThreadTeam  team1(3, 1, "TestRunningClosedWorkPub.log");
    ThreadTeam  team2(4, 2, "TestRunningClosedWorkPub.log");

    team1.attachWorkReceiver(&team2);

    for (unsigned int i=0; i<N_ITERS; ++i) {
        // Team 1 needs to delay long enough that we can catch transitions, 
        // but fast enough that all three units of work will be handled by
        // team 2 simulataneously
        team1.startTask(TestThreadRoutines::delay_10ms,  1, "quick",  "10ms");
        team2.startTask(TestThreadRoutines::delay_100ms, 4, "wait",   "100ms");
        team1.enqueue(1);
        team1.enqueue(2);
        team1.enqueue(3);
        team1.closeTask();
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)     break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(2, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(4, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // When the first work is finished, it should be enqueued
        // on team 2
        for (unsigned int i=0; i<50; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (unsigned int i=0; i<20; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(1, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(3, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        // When the second work is finished, it should be enqueued
        // on team 2
        for (unsigned int i=0; i<50; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 0)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (unsigned int i=0; i<20; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 2)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(2, N_comp);
        EXPECT_EQ(0, N_Q);

        // When the final work is finished, it should be enqueued
        // on team 2
        team1.wait();

        // Wait until computing on final unit begins *and* the single waiting
        // thread goes idle
        for (unsigned int i=0; i<20; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_idle == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(3, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.wait();
    }

    team1.detachWorkReceiver();
}

/**
 * Confirm that ThreadTeam interface methods that should fail in the Running &
 * No More Work mode do fail.
 */
TEST(ThreadTeamTest, TestRunningNoMoreWorkErrors) {
    unsigned int  N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(5, 1, "TestRunningNoMoreWorkErrors.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        // Team must have at least one thread and one units of work to stay in
        // Running & No More Work
        team1.startTask(TestThreadRoutines::delay_100ms, 1, "wait",  "100ms");
        team1.enqueue(1);
        team1.closeTask();

        EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team1.mode());
        ASSERT_THROW(team1.startTask(TestThreadRoutines::noop, 0,
                                     "quick",  "fail"), std::runtime_error);
        ASSERT_THROW(team1.startTask(TestThreadRoutines::noop, 1,
                                     "quick",  "fail"), std::runtime_error);

        ASSERT_THROW(team1.increaseThreadCount(0), std::logic_error);
        ASSERT_THROW(team1.increaseThreadCount(team1.nMaximumThreads()+1),
                                               std::logic_error);

        ASSERT_THROW(team1.enqueue(1),  std::runtime_error);
        ASSERT_THROW(team1.closeTask(), std::runtime_error);

        // Make certain that all of the above was still done in Running & Closed
        EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team1.mode());

        team1.wait();
    }
}

/**
 * 
 */
TEST(ThreadTeamTest, TestRunningNoMoreWorkForward) {
    unsigned int  N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam  team1(2, 1, "TestRunningNoMoreWorkForward.log");
    ThreadTeam  team2(3, 1, "TestRunningNoMoreWorkForward.log");

    team1.attachThreadReceiver(&team2);

    // Team 1 shall be set into Running & No More Work with a thread
    //        carrying out a lengthy computation
    // Team 2 shall be in Running & Open with a pending unit of work
    //        but no threads
    team1.startTask(TestThreadRoutines::delay_100ms, 1, "wait", "100ms");
    team2.startTask(TestThreadRoutines::delay_100ms, 0, "wait", "100ms");
    team1.enqueue(1);
    team2.enqueue(2);
    team1.closeTask();
    for (unsigned int i=0; i<10; ++i) {
        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_comp == 1)     break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team1.mode());
    EXPECT_EQ(2, team1.nMaximumThreads());
    EXPECT_EQ(1, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(1, N_comp);
    EXPECT_EQ(0, N_Q);

    team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
    EXPECT_EQ(3, team2.nMaximumThreads());
    EXPECT_EQ(3, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(1, N_Q);

    // Team 1 doesn't need another thread and should forward it to
    // team 2
    team1.increaseThreadCount(1);

    for (unsigned int i=0; i<10; ++i) {
        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_comp == 1)     break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_NO_MORE_WORK, team1.mode());
    EXPECT_EQ(2, team1.nMaximumThreads());
    EXPECT_EQ(1, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(1, N_comp);
    EXPECT_EQ(0, N_Q);
   
    // If we didn't call increaseThreadCount above, then team 1 still
    // hasn't finished its task and wouldn't have forwarded its 1 thread along
    // yet.  This would fail.
    team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeam::MODE_RUNNING_OPEN_QUEUE, team2.mode());
    EXPECT_EQ(3, team2.nMaximumThreads());
    EXPECT_EQ(2, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(1, N_comp);
    EXPECT_EQ(0, N_Q);

    team1.wait();

    team2.closeTask();
    team2.wait();

    team1.detachThreadReceiver();
}

}

