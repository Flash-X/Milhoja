/*******************************************************************************
*
*  IMPORTANT: If changes are made to tests, then the person making the changes
*  must also update the test coverage information in RuntimeDesign.xlsx.
*
*******************************************************************************/

#include <cmath>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <pthread.h>

#include "estimateTimerResolution.h"

#include "threadTeamTest.h"
#include "testThreadRoutines.h"
#include "RuntimeAction.h"
#include "ThreadTeam.h"

#include "gtest/gtest.h"

using namespace orchestration;

namespace {

void processWalltimes(const std::vector<double>& wtimes, 
                      double* mean_wtime, double* std_wtime) {
    unsigned int   N = wtimes.size();

    // Estimate mean wtime with sample mean
    double sum = 0.0;
    for (auto& wt : wtimes) {
        sum += wt;
    }
    double mean = sum / static_cast<double>(N);

    // Estimate standard deviation with sample variance
    sum = 0.0;
    for (auto& wt : wtimes) {
        sum += (wt - mean) * (wt - mean);
    }

    *mean_wtime = mean;
    *std_wtime = sqrt(sum / static_cast<double>(N-1));
}

/**
 *   Define a test fixture
 */ 
class ThreadTeamTest : public testing::Test {
protected:
    RuntimeAction    nullRoutine;
    RuntimeAction    noop;
    RuntimeAction    delay_10ms;
    RuntimeAction    delay_100ms;

    ThreadTeamTest(void) {
        nullRoutine.name = "nullRoutine";
        nullRoutine.nInitialThreads = 0;
        nullRoutine.teamType = ThreadTeamDataType::BLOCK;
        nullRoutine.routine = nullptr;

        noop.name = "noop";
        noop.nInitialThreads = 0;
        noop.teamType = ThreadTeamDataType::BLOCK;
        noop.routine = TestThreadRoutines::noop;

        delay_10ms.name = "10ms";
        delay_10ms.nInitialThreads = 0;
        delay_10ms.teamType = ThreadTeamDataType::BLOCK;
        delay_10ms.routine = TestThreadRoutines::delay_10ms;

        delay_100ms.name = "100ms";
        delay_100ms.nInitialThreads = 0;
        delay_100ms.teamType = ThreadTeamDataType::BLOCK;
        delay_100ms.routine = TestThreadRoutines::delay_100ms;
    }

    ~ThreadTeamTest(void) { }
};

/**
 *  Test that teams are created and start in the appropriate initial state.
 *  This implicitly tests destruction from the Idle mode.
 */
TEST_F(ThreadTeamTest, TestInitialState) {
    unsigned int   N_ITERS = 100;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    // Check that having two thread teams available at the same time
    // is OK in terms of initial state
    ThreadTeam<int>   team1(10, 1, "TestInitialState.log");
    ThreadTeam<int>*  team2 = nullptr;

    // Confirm explicit state
    team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(10, team1.nMaximumThreads());
    EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
    EXPECT_EQ(10, N_idle);
    EXPECT_EQ(0,  N_wait);
    EXPECT_EQ(0,  N_comp);
    EXPECT_EQ(0,  N_Q);

    // Confirm indirectly that no subscribers are attached
    EXPECT_THROW(team1.detachThreadReceiver(), std::logic_error);
    EXPECT_THROW(team1.detachDataReceiver(),   std::logic_error);

    // Check that teams must have minimum number of threads
    EXPECT_THROW(new ThreadTeam<int>(0, 2, "TestInitialState.log"), std::logic_error);
    EXPECT_THROW(new ThreadTeam<int>(1, 2, "TestInitialState.log"), std::logic_error);

    for (unsigned int i=2; i<=N_ITERS; ++i) {
        team2 = new ThreadTeam<int>(i, 2, "TestInitialState.log");

        team2->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(i, team2->nMaximumThreads());
        EXPECT_EQ(ThreadTeamMode::IDLE, team2->mode());
        EXPECT_EQ(i, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        EXPECT_THROW(team2->detachThreadReceiver(), std::logic_error);
        EXPECT_THROW(team2->detachDataReceiver(),   std::logic_error);

        delete team2;
        team2 = nullptr;
    }
}

/**
 * Test if a ThreadTeam object is destroyed correctly even if the mode isn't in
 * Idle (i.e. there are at least some threads in the Computing and Wait states).
 *
 * As presently implemented, this cannot be tested automatically.  Rather, we
 * should not see an error reports in the test output and we need to manually
 * study the log file to confirm that the destruction happened as expected.
 *
 * \todo - figure out how to test this automatically
 */
TEST_F(ThreadTeamTest, TestDestruction) {
    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int  work = 1;

    ThreadTeam<int>*    team1 = new ThreadTeam<int>(4, 1, "TestDestruction.log");
    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(ThreadTeamMode::IDLE, team1->mode());

    // Test in Running & Open 
    delay_100ms.nInitialThreads = 3;
    team1->startCycle(delay_100ms, "teamName");
    team1->enqueue(work, false);
    for (unsigned int i=0; i<10; ++i) {
        team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if ((N_comp == 1) && (N_wait == 2))      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1->mode());
    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(1, N_idle);
    EXPECT_EQ(2, N_wait);
    EXPECT_EQ(1, N_comp);
    EXPECT_EQ(0, N_Q);

    // Set to terminate
    delete  team1;
    team1 = nullptr;
 
    team1 = new ThreadTeam<int>(4, 1, "TestDestruction.log");
    EXPECT_EQ(4, team1->nMaximumThreads());
    EXPECT_EQ(ThreadTeamMode::IDLE, team1->mode());
 
    // Wait for a thread to dequeue the work before closing queue
    // The mode will transition to Running & No More Work as there
    // won't be any work in the queue
    delay_100ms.nInitialThreads = 3;
    team1->startCycle(delay_100ms, "teamName");
    team1->enqueue(work, false);
    for (unsigned int i=0; i<10; ++i) {
        team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_Q == 0)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Close task & let Waiting threads go Idle
    team1->closeQueue();
    for (unsigned int i=0; i<100; ++i) {
        team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_idle == 3)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team1->stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1->mode());
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
TEST_F(ThreadTeamTest, TestIdleWait) {
    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam<int>    team1(3, 1, "TestIdleWait.log");

    // Call wait without having run a task
    EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
    team1.wait();
    team1.wait();
    team1.wait();

    // Do an execution cycle with no threads/no work and call wait 
    // as many times as we want
    noop.nInitialThreads = 0;
    team1.startCycle(noop, "test1");
    team1.closeQueue();
    for (unsigned int i=0; i<10; ++i) {
        if (team1.mode() == ThreadTeamMode::IDLE)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
    team1.wait();
    team1.wait();
    team1.wait();

    // We want the wait to be called before work finishes and the team
    // transitions to Idle
    delay_10ms.nInitialThreads = 1;
    team1.startCycle(delay_10ms, "test1");
    int work = 1;
    team1.enqueue(work, false);

    for (unsigned int i=0; i<10; ++i) {
        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        if (N_comp == 1)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    team1.closeQueue();
    // Legitimate wait call
    team1.wait();
    EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());

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
TEST_F(ThreadTeamTest, TestNoWorkNoThreads) {
    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam<int>    team1(3, 1, "TestNoWorkNoThreads.log");
    ThreadTeam<int>    team2(2, 2, "TestNoWorkNoThreads.log");

    team1.attachDataReceiver(&team2);

    EXPECT_EQ(3, team1.nMaximumThreads());
    EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());

    EXPECT_EQ(2, team2.nMaximumThreads());
    EXPECT_EQ(ThreadTeamMode::IDLE, team2.mode());

    noop.nInitialThreads = 0;
    team1.startCycle(noop, "test1");
    team2.startCycle(noop, "test2");

    team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
    EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
    EXPECT_EQ(3, team1.nMaximumThreads());
    EXPECT_EQ(3, N_idle);
    EXPECT_EQ(0, N_wait);
    EXPECT_EQ(0, N_comp);
    EXPECT_EQ(0, N_Q);

    team1.closeQueue();
    team1.wait();
    // Next line will hang if team1 doesn't call team2's closeQueue()
    team2.wait();

    EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
    EXPECT_EQ(ThreadTeamMode::IDLE, team2.mode());
}

/**
 * Confirm that the thread team responds to incorrect use of team's interface in
 * the Idle mode.
 */
TEST_F(ThreadTeamTest, TestIdleErrors) {
    unsigned int   N_ITERS = 10;

    int work = 1;

    ThreadTeam<int>    team1(10, 1, "TestIdleErrors.log");
    ThreadTeam<int>    team2(5,  2, "TestIdleErrors.log");
    ThreadTeam<int>    team3(2,  3, "TestIdleErrors.log");
    ThreadTeam<int>    team4(2,  4, "TestIdleErrors.log");

    for (unsigned int i=0; i<N_ITERS; ++i) {
        // No task given
        nullRoutine.nInitialThreads = team1.nMaximumThreads() - 1;
        EXPECT_THROW(team1.startCycle(nullRoutine, "teamName"),
                     std::logic_error);

        // Ask for more threads than in team
        noop.nInitialThreads = team1.nMaximumThreads() + 1;
        EXPECT_THROW(team1.startCycle(noop, "teamName"),
                     std::logic_error);
        EXPECT_THROW(team1.increaseThreadCount(team1.nMaximumThreads()+1), std::logic_error);

        // No sense in increasing threads by zero
        EXPECT_THROW(team1.increaseThreadCount(0),  std::logic_error);

        // Use methods that are not allowed in Idle
        EXPECT_THROW(team1.enqueue(work, true),  std::runtime_error);
        EXPECT_THROW(team1.enqueue(work, false), std::runtime_error);
        EXPECT_THROW(team1.closeQueue(), std::runtime_error);

        // Detach when no teams have been attached
        EXPECT_THROW(team1.detachThreadReceiver(), std::logic_error);
        EXPECT_THROW(team1.detachDataReceiver(),   std::logic_error);

        // Attach null team
        EXPECT_THROW(team1.attachThreadReceiver(nullptr), std::logic_error);
        EXPECT_THROW(team1.attachDataReceiver(nullptr),   std::logic_error);

        // Attach team to itself
        EXPECT_THROW(team1.attachThreadReceiver(&team1), std::logic_error);
        EXPECT_THROW(team1.attachDataReceiver(&team1),   std::logic_error);

        // Setup basic thread team configuration
        team1.attachThreadReceiver(&team3);
        team2.attachThreadReceiver(&team3);
        team2.attachDataReceiver(&team3);

        // Not allowed to attach more than one receiver
        //  - try with same and different teams
        EXPECT_THROW(team1.attachThreadReceiver(&team3), std::logic_error);
        EXPECT_THROW(team2.attachThreadReceiver(&team3), std::logic_error);
        EXPECT_THROW(team2.attachDataReceiver(&team3),   std::logic_error);

        EXPECT_THROW(team1.attachThreadReceiver(&team4), std::logic_error);
        EXPECT_THROW(team2.attachThreadReceiver(&team4), std::logic_error);
        EXPECT_THROW(team2.attachDataReceiver(&team4),   std::logic_error);

        // Break down configuration so that destruction is clean
        team1.detachThreadReceiver();
        team2.detachThreadReceiver();
        team2.detachDataReceiver();

        // If these were properly detached above, these should fail
        EXPECT_THROW(team1.detachThreadReceiver(), std::logic_error);
        EXPECT_THROW(team2.detachThreadReceiver(), std::logic_error);
        EXPECT_THROW(team1.detachDataReceiver(),   std::logic_error);
        EXPECT_THROW(team2.detachDataReceiver(),   std::logic_error);

        // Run an execution cycle so that we test the above with and without
        // having run a cycle
        noop.nInitialThreads = 5;
        team1.startCycle(noop, "test1");
        team1.enqueue(work, false);
        team1.closeQueue();
        team1.wait();
    }
}

/**
 *  Confirm that threads are forwarded properly by a team that is in Idle
 *  and that a team in Running & Closed only uses those threads it needs and
 *  forwards along the rest.
 */ 
TEST_F(ThreadTeamTest, TestIdleForwardsThreads) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 0;

    ThreadTeam<int>  team1(2, 1, "TestIdleForwardsThreads.log");
    ThreadTeam<int>  team2(4, 2, "TestIdleForwardsThreads.log");
    ThreadTeam<int>  team3(6, 3, "TestIdleForwardsThreads.log");

    // Team 2 is a thread subscriber and publisher
    team1.attachThreadReceiver(&team2);
    team2.attachThreadReceiver(&team3);

    for (unsigned int i=0; i<N_ITERS; ++i) {
        // Team 1 stays in Idle
        // Team 2 setup in Running & Closed with one pending item
        //        It will take Team 2 some time to finish its 
        //        work once it gets a thread
        // Team 3 setup in Running & Closed with one pending item
        //        It will finish quickly once it gets a thread
        noop.nInitialThreads = 0;
        delay_100ms.nInitialThreads = 0;
        team2.startCycle(delay_100ms, "wait");
        team3.startCycle(noop,        "quick");
        work = 1;
        team2.enqueue(work, false);
        work = 2;
        team3.enqueue(work, false);
        team2.closeQueue();
        team3.closeQueue();

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
        EXPECT_EQ(2, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(4, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(1, N_Q);

        team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team3.mode());
        EXPECT_EQ(6, team3.nMaximumThreads());
        EXPECT_EQ(6, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(1, N_Q);

        // Team 1 should forward 2 threads to Team 2
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
        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::IDLE, team3.mode());
        EXPECT_EQ(6, team3.nMaximumThreads());
        EXPECT_EQ(6, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.wait();

        EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
        EXPECT_EQ(ThreadTeamMode::IDLE, team2.mode());
        EXPECT_EQ(ThreadTeamMode::IDLE, team3.mode());
    }
}

/**
 * Confirm proper functionality when no work is given, but we still run cycles.
 */
TEST_F(ThreadTeamTest, TestNoWork) {
    unsigned int   N_ITERS = 100;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam<int>  team1(5, 1, "TestNoWork.log");

    noop.nInitialThreads = 3;
    for (unsigned int i=0; i<N_ITERS; ++i) {
        team1.startCycle(noop, "test");
        for (unsigned int i=0; i<100; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 3)  break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(5, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(3, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team1.closeQueue();
        team1.wait();

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
        EXPECT_EQ(5, team1.nMaximumThreads());
        EXPECT_EQ(5, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);
    }
}

/**
 * Confirm that the thread team responds to incorrect use of team's interface in
 * the Running & Open mode.
 */
TEST_F(ThreadTeamTest, TestRunningOpenErrors) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    ThreadTeam<int>  team1(10, 1, "TestRunningOpenErrors.log");
    ThreadTeam<int>  team2(10, 2, "TestRunningOpenErrors.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        noop.nInitialThreads = 5;
        team1.startCycle(noop, "test");
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 5)     break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(10, team1.nMaximumThreads());
        EXPECT_EQ(5, N_idle);
        EXPECT_EQ(5, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Call methods that are not allowed in Running & Open
        nullRoutine.nInitialThreads = 0;
        EXPECT_THROW(team1.startCycle(nullRoutine, "test"),
                     std::logic_error);

        noop.nInitialThreads = 0;
        EXPECT_THROW(team1.startCycle(noop, "test"),
                     std::runtime_error);

        EXPECT_THROW(team1.attachThreadReceiver(&team2), std::logic_error);
        EXPECT_THROW(team1.attachDataReceiver(&team2),   std::logic_error);

        EXPECT_THROW(team1.increaseThreadCount(0),  std::logic_error);
        EXPECT_THROW(team1.increaseThreadCount(team1.nMaximumThreads()+1),
                     std::logic_error);

        // Confirm that all of the above were called in the same mode
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());

        team1.closeQueue();
        team1.wait();
    }

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        team1.attachThreadReceiver(&team2);
        team1.attachDataReceiver(&team2);

        noop.nInitialThreads = 5;
        team1.startCycle(noop, "quick1");
        noop.nInitialThreads = 0;
        team2.startCycle(noop, "quick2");
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 5)     break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(10, team1.nMaximumThreads());
        EXPECT_EQ(5, N_idle);
        EXPECT_EQ(5, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        EXPECT_THROW(team1.detachThreadReceiver(), std::logic_error);
        EXPECT_THROW(team1.detachDataReceiver(),   std::logic_error);

        // Confirm that all of the above were called in the same mode
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());

        team1.closeQueue();
        team1.wait();
        team2.wait();

        team1.detachThreadReceiver();
        team1.detachDataReceiver();
    }
}

/**
 *  Confirm that threads can be increased correctly in Running & Open.
 */
TEST_F(ThreadTeamTest, TestRunningOpenIncreaseThreads) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 1;

    ThreadTeam<int>  team1(3, 1, "TestRunningOpenIncreaseThreads.log");
    ThreadTeam<int>  team2(4, 2, "TestRunningOpenIncreaseThreads.log");

    team1.attachThreadReceiver(&team2);

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        // All teams should generally have startCycle called before we start
        // interacting with any team (e.g. increasing threads or adding work)
        noop.nInitialThreads = 0;
        team1.startCycle(noop, "test1");
        team2.startCycle(noop, "test2");
        team1.enqueue(work, false);

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(1, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(4, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Increasing count in Open means that Team 1 should keep all that we
        // have given even though we have more waiting threads than work in the
        // queue
        team1.increaseThreadCount(2);
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if ((N_Q == 0) && (N_wait == 2))   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(4, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // This should result in threads being sent to Team 2
        team1.closeQueue();
        team1.wait();
        for (unsigned int i=0; i<10; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 2)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.closeQueue();
        team2.wait();
    }

    team1.detachThreadReceiver();
}

/**
 *  Confirm that calling enqueue with all non-Idle threads in Wait results in a
 *  Wait thread transitioning to a Compute thread.  Also make certain that
 *  computing threads push work to Work subscribers.
 */
TEST_F(ThreadTeamTest, TestRunningOpenEnqueue) {
    unsigned int   N_ITERS = 2;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 0;

    ThreadTeam<int>  team1(3, 1, "TestRunningOpenEnqueue.log");
    ThreadTeam<int>  team2(2, 2, "TestRunningOpenEnqueue.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        team1.attachDataReceiver(&team2);

        delay_100ms.nInitialThreads = 1;
        team1.startCycle(delay_100ms, "wait1");
        delay_100ms.nInitialThreads = 2;
        team2.startCycle(delay_100ms, "wait2");
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 1)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(1, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Enqueue two units of work so that the single thread will necessarily
        // finish work on one, emit/receive computationFinished and find the
        // next unit of work
        work = 1;
        team1.enqueue(work, false);
        work = 2;
        team1.enqueue(work, false);
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(1, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(2, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Wait until Team 2 gets work from Team 1
        for (unsigned int i=0; i<110; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)   break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Working on second unit of work now
        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        // Confirm that team 2 got a unit of work
        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(2, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(1, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        // Thread team 1 will call closeQueue for team 2
        team1.closeQueue();
        team1.wait();
        team2.wait();

        team1.detachDataReceiver();
    }
}

/**
 * Confirm that ThreadTeam interface methods that should fail in the Running &
 * Closed mode do fail.
 */
TEST_F(ThreadTeamTest, TestRunningClosedErrors) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 0;

    ThreadTeam<int>  team1(5, 1, "TestRunningClosedErrors.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        // Team has one active thread and two units of work to stay closed
        delay_100ms.nInitialThreads = 1;
        team1.startCycle(delay_100ms, "wait");
        work = 1;
        team1.enqueue(work, false);
        work = 2;
        team1.enqueue(work, false);
        team1.closeQueue();

        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team1.mode());
        nullRoutine.nInitialThreads = 0;
        EXPECT_THROW(team1.startCycle(nullRoutine, "fail"), std::logic_error);
        noop.nInitialThreads = 0;
        EXPECT_THROW(team1.startCycle(noop, "fail"), std::runtime_error);
        noop.nInitialThreads = 1;
        EXPECT_THROW(team1.startCycle(noop, "fail"), std::runtime_error);

        EXPECT_THROW(team1.increaseThreadCount(0), std::logic_error);
        EXPECT_THROW(team1.increaseThreadCount(team1.nMaximumThreads()+1),
                                               std::logic_error);

        work = 1;
        EXPECT_THROW(team1.enqueue(work, true),  std::runtime_error);
        EXPECT_THROW(team1.enqueue(work, false), std::runtime_error);
        EXPECT_THROW(team1.closeQueue(), std::runtime_error);

        // Make certain that all of the above was still done in Running & Closed
        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team1.mode());

        team1.wait();
    }
}

/**
 * Confirm that activating threads when in Running & Closed works as expected.
 */
TEST_F(ThreadTeamTest, TestRunningClosedActivation) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 0;

    ThreadTeam<int>  team1(4, 1, "TestRunningClosedActivation.log");

    for (unsigned int i=0; i<N_ITERS; ++i) {
        // Add enough work to test all necessary transitions and wait until the
        // single thread starts computing
        delay_100ms.nInitialThreads = 1;
        team1.startCycle(delay_100ms, "wait");
        work = 1;
        team1.enqueue(work, false);
        work = 2;
        team1.enqueue(work, false);
        work = 3;
        team1.enqueue(work, false);
        team1.closeQueue();
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 2)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(4, team1.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(2, N_Q);

        // Activate a thread so that it can start computing 
        // This shouldn't transition the mode as there will still be a unit of
        // work in the queue after this transition
        team1.increaseThreadCount(1);
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_Q == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(4, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
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

        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1.mode());
        EXPECT_EQ(4, team1.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(3, N_comp);
        EXPECT_EQ(0, N_Q);

        team1.wait();
    }
}

/**
 * Confirm that activating threads when in Running & Closed works as expected.
 */
TEST_F(ThreadTeamTest, TestRunningClosedWorkPub) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 0;

    ThreadTeam<int>  team1(3, 1, "TestRunningClosedWorkPub.log");
    ThreadTeam<int>  team2(4, 2, "TestRunningClosedWorkPub.log");

    team1.attachDataReceiver(&team2);

    for (unsigned int i=0; i<N_ITERS; ++i) {
        // Team 1 needs to delay long enough that we can catch transitions, 
        // but fast enough that all three units of work will be handled by
        // team 2 simultaneously.  Similarly, we activate all threads in team 2
        // from the start
        delay_10ms.nInitialThreads = 1;
        team1.startCycle(delay_10ms,  "quick");
        delay_100ms.nInitialThreads = 4;
        team2.startCycle(delay_100ms, "wait");
        work = 1;
        team1.enqueue(work, false);
        work = 2;
        team1.enqueue(work, false);
        work = 3;
        team1.enqueue(work, false);
        team1.closeQueue();
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)    break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        for (unsigned int i=0; i<10; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 4)    break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(2, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(4, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // When the first work unit is finished, it should be enqueued
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
        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(1, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
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
        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1.mode());
        EXPECT_EQ(3, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(0, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(2, N_comp);
        EXPECT_EQ(0, N_Q);

        // When the final work is finished, it should be enqueued
        // on team 2 and team 1 should call closeQueue for team 2
        team1.wait();

        // Wait until computing on final unit begins *and* the single waiting
        // thread goes idle due to the transition to Running & No More Work
        for (unsigned int i=0; i<20; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_idle == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team2.mode());
        EXPECT_EQ(4, team2.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(3, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.wait();
    }

    team1.detachDataReceiver();
}

/**
 * Confirm that ThreadTeam interface methods that should fail in the Running &
 * No More Work mode do fail.
 */
TEST_F(ThreadTeamTest, TestRunningNoMoreWorkErrors) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 1;

    ThreadTeam<int>  team1(5, 1, "TestRunningNoMoreWorkErrors.log");

    for (unsigned int i=0; i<N_ITERS; ++i) { 
        // Team must have at least one unit of work to stay in
        // Running & No More Work
        delay_100ms.nInitialThreads = 1;
        team1.startCycle(delay_100ms, "wait");
        team1.enqueue(work, false);
        team1.closeQueue();
        for (unsigned int i=0; i<50; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1.mode());
        nullRoutine.nInitialThreads = 0;
        EXPECT_THROW(team1.startCycle(nullRoutine, "null"), std::logic_error);
        noop.nInitialThreads = 0;
        EXPECT_THROW(team1.startCycle(noop, "fail"), std::runtime_error);
        noop.nInitialThreads = 1;
        EXPECT_THROW(team1.startCycle(noop, "fail"), std::runtime_error);

        EXPECT_THROW(team1.increaseThreadCount(0), std::logic_error);
        EXPECT_THROW(team1.increaseThreadCount(team1.nMaximumThreads()+1),
                                               std::logic_error);

        EXPECT_THROW(team1.enqueue(work, true),  std::runtime_error);
        EXPECT_THROW(team1.enqueue(work, false), std::runtime_error);
        EXPECT_THROW(team1.closeQueue(), std::runtime_error);

        // Make certain that all of the above was still done in Running & Closed
        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1.mode());

        team1.wait();
    }
}

/**
 *  Confirm that in Running & No More Work all attempts to increase the thread
 *  count are forwarded on to the thread subscriber.
 */
TEST_F(ThreadTeamTest, TestRunningNoMoreWorkForward) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 1;

    ThreadTeam<int>  team1(2, 1, "TestRunningNoMoreWorkForward.log");
    ThreadTeam<int>  team2(3, 2, "TestRunningNoMoreWorkForward.log");

    team1.attachThreadReceiver(&team2);

    for (unsigned int i=0; i<N_ITERS; ++i) {
        // Team 1 shall be set into Running & No More Work with a thread
        //        carrying out a lengthy computation.  We set the team
        //        up so that there is an idle thread that could be activated
        // Team 2 shall be in Running & Open with a pending unit of work
        //        but no threads
        delay_100ms.nInitialThreads = 1;
        team1.startCycle(delay_100ms, "wait");
        delay_100ms.nInitialThreads = 0;
        team2.startCycle(delay_100ms, "wait");
        work = 1;
        team1.enqueue(work, false);
        work = 2;
        team2.enqueue(work, false);
        team1.closeQueue();
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_comp == 1)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1.mode());
        EXPECT_EQ(2, team1.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
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

        // Idle thread remained idle
        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1.mode());
        EXPECT_EQ(2, team1.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        // If we didn't call increaseThreadCount above, then team 1 still
        // hasn't finished its task and wouldn't have forwarded its 1 thread along
        // yet.  This would fail.
        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(3, team2.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(1, N_comp);
        EXPECT_EQ(0, N_Q);

        // This should send all of Team 1's active threads to Team 2
        team1.wait();

        // Wait for team 2 computation to finish so that we have a 
        // predictable state to check
        for (unsigned int i=0; i<10; ++i) {
            team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 2)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(3, team2.nMaximumThreads());
        EXPECT_EQ(1, N_idle);
        EXPECT_EQ(2, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.closeQueue();
        team2.wait();
    }

    team1.detachThreadReceiver();
}

/**
 * Confirm that the transitionThread and computationFinished threads are handled
 * correctly in the Running & No More Work mode
 */
TEST_F(ThreadTeamTest, TestRunningNoMoreWorkTransition) {
    unsigned int   N_ITERS = 10;

    unsigned int   N_idle = 0;
    unsigned int   N_wait = 0;
    unsigned int   N_comp = 0;
    unsigned int   N_Q    = 0;

    int work = 0;

    ThreadTeam<int>  team1(8, 1, "TestRunningNoMoreWorkTransition.log");
    ThreadTeam<int>  team2(3, 2, "TestRunningNoMoreWorkTransition.log");
    ThreadTeam<int>  team3(8, 3, "TestRunningNoMoreWorkTransition.log");

    team1.attachDataReceiver(&team2);
    team1.attachThreadReceiver(&team3);

    for (unsigned int i=0; i<N_ITERS; ++i) {
        // Queue up several units of work so that we can confirm the proper behavior
        // when computationFinished is emitted in Running & No More Work
        //
        // Give the task enough threads that at least two threads will be waiting so
        // that we can confirm the proper behavior when transitionThread is emitted
        // in Running & No More Work
        //
        // Setup Team 3 without any initial threads and no work to confirm that 
        // thread forwarding progresses when a thread transitions to Idle.
        delay_100ms.nInitialThreads = 6;
        noop.nInitialThreads = 0;
        team1.startCycle(delay_100ms, "wait1");
        team2.startCycle(noop,        "quick2");
        team3.startCycle(noop,        "quick3");
        work = 1;
        team1.enqueue(work, false);
        work = 2;
        team1.enqueue(work, false);
        work = 3;
        team1.enqueue(work, false);
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if ((N_comp == 3) && (N_wait == 3))     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Confirm that we have at least two threads waiting
        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team1.mode());
        EXPECT_EQ(8, team1.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(3, N_wait);
        EXPECT_EQ(3, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(3, team2.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team3.mode());
        EXPECT_EQ(8, team3.nMaximumThreads());
        EXPECT_EQ(8, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Calling closeQueue in this state should result in the broadcast
        // of transitionThread to all waiting threads
        // Confirm here that the waiting threads are correctly transitioned to Idle
        // and that thread resources are forwarded to Team 3
        team1.closeQueue();
        for (unsigned int i=0; i<10; ++i) {
            team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_idle == 5)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        for (unsigned int i=0; i<10; ++i) {
            team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 3)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_NO_MORE_WORK, team1.mode());
        EXPECT_EQ(8, team1.nMaximumThreads());
        EXPECT_EQ(5, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(3, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team2.mode());
        EXPECT_EQ(3, team2.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team3.mode());
        EXPECT_EQ(8, team3.nMaximumThreads());
        EXPECT_EQ(5, N_idle);
        EXPECT_EQ(3, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // As work is finished by Team 1, it should be forwarded to Team 2.
        // Also, when Team 1 finishes its work, it should unblock this wait
        // and call closeQueue on Team 2.  This should transition the mode of Team 2.
        // Finally, the remaining thread resources should be transferred to
        // Team 3
        team1.wait();
        for (unsigned int i=0; i<10; ++i) {
            team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
            if (N_wait == 6)     break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Confirm that computing threads transitioned to Idle correctly
        team1.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::IDLE, team1.mode());
        EXPECT_EQ(8, team1.nMaximumThreads());
        EXPECT_EQ(8, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        // Confirm that work was forwarded correctly
        team2.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_CLOSED_QUEUE, team2.mode());
        EXPECT_EQ(3, team2.nMaximumThreads());
        EXPECT_EQ(3, N_idle);
        EXPECT_EQ(0, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(3, N_Q);

        // Confirm that threads forwarded correctly
        team3.stateCounts(&N_idle, &N_wait, &N_comp, &N_Q);
        EXPECT_EQ(ThreadTeamMode::RUNNING_OPEN_QUEUE, team3.mode());
        EXPECT_EQ(8, team3.nMaximumThreads());
        EXPECT_EQ(2, N_idle);
        EXPECT_EQ(6, N_wait);
        EXPECT_EQ(0, N_comp);
        EXPECT_EQ(0, N_Q);

        team2.increaseThreadCount(3);
        team2.wait();
        team3.closeQueue();
        team3.wait();
    }

    team1.detachDataReceiver();
    team1.detachThreadReceiver();
}

#ifndef DEBUG_RUNTIME
TEST_F(ThreadTeamTest, TestTimings) {
    using namespace std::chrono;

    using microseconds = std::chrono::duration<double, std::micro>;

    unsigned int   N_ITERS = 5000;
    unsigned int   N_WORK  = 100;

    int work = 1;

    std::vector<double>  wtimes_us(N_ITERS);
    auto                 start = steady_clock::now();
    auto                 end   = steady_clock::now();
    double               mean_wtime_us = 0.0;
    double               std_wtime_us = 0.0;
    std::ofstream        fptr;
    std::string          filename;

    double resolution = 0.0; 
    std::string  resolution_str;
    try {
        resolution = estimateSteadyClockResolution();
        resolution_str = std::to_string(resolution) + " us";
    } catch (std::runtime_error&) {
        resolution = -1.0;
        resolution_str = "Too small to measure";
    }

    ThreadTeam<int>* team1 = nullptr;
    ThreadTeam<int>  team2(T3::nThreadsPerTeam, 2, "TestTimings2.log");

    std::cout << "\nTiming using C++ standard library steady clock\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Is steady\t\t\t"
              << (steady_clock::is_steady ? "T" : "F") << "\n";
    std::cout << "Is duration FP\t\t\t"
              << (std::chrono::treat_as_floating_point<microseconds::rep>::value ? "T" : "F")
              << "\n";
    std::cout << "Minimum duration\t\t"
              << steady_clock::duration::min().count() << "\n"; 
    std::cout << "Maximum duration\t\t"
              << steady_clock::duration::max().count() << "\n"; 
    std::cout << "Measured Clock resolution\t" << resolution_str << "\n";
    std::cout << "Clock unit\t\t\t"
              <<                       std::chrono::steady_clock::period::num
                 / static_cast<double>(std::chrono::steady_clock::period::den)
              << " s\n\n";
    std::cout << "NOTE: All timing data was collected under the assumption that each time\n"
              << "      is much larger than the above clock resolution.\n" << std::endl; 

    //***** RUN CREATION TIME EXPERIMENT
    for (unsigned int i=0; i<wtimes_us.size(); ++i) {
        start = steady_clock::now();
        team1 = new ThreadTeam<int>(T3::nThreadsPerTeam, 1, "TestTimings1.log");
        end = steady_clock::now();
        wtimes_us[i] = microseconds(end - start).count();
        EXPECT_TRUE(wtimes_us[i] > 0.0);

        delete team1;
        team1 = nullptr;
    }
    processWalltimes(wtimes_us, &mean_wtime_us, &std_wtime_us);
    std::cout << "Thread Team Create Time\t\t\t\t\t\t\t"
              << mean_wtime_us << " +/- "
              << std_wtime_us  << " us\n";

    //***** PROBE INFORMATION ON Idle->Wait TRANSITION TIMES
    for (unsigned int n=0; n<=T3::nThreadsPerTeam; ++n) {
        for (unsigned int i=0; i<wtimes_us.size(); ++i) {
            start = steady_clock::now();
            // By passing true, this should not return until all N_thread
            // threads are in Wait
            noop.nInitialThreads = n;
            team2.startCycle(noop, "quick", true);
            end = steady_clock::now();
            wtimes_us[i] = microseconds(end - start).count();
            EXPECT_TRUE(wtimes_us[i] > 0.0);

            team2.closeQueue();
            team2.wait();
        }
        processWalltimes(wtimes_us, &mean_wtime_us, &std_wtime_us);
        std::cout << n << " thread/"
                  << 0 << " units of work enqueued/"
                  << "startCycle Time/ wait=T\t\t" 
                  << mean_wtime_us << " +/- "
                  << std_wtime_us  << " us\n";

        filename =   "StartCycleTimings_"
                   + std::to_string(n)
                   + "_threads.dat";
        fptr.open(filename, std::ios::out);
        fptr << "# C++ steady_clock\n";
        fptr << "# Clock Resolution\t\t" << resolution_str << "\n";
        fptr << "N_threads,N_work,wtime_us\n";
        for (auto& wt_us : wtimes_us) {
            fptr << std::setprecision(15)
                 << n << ","
                 << 0 << ","
                 << wt_us << "\n";
        }
        fptr.close();
    }

    //***** PROBE INFORMATION ON Wait->Idle TRANSITION TIMES
    for (unsigned int n=0; n<=T3::nThreadsPerTeam; ++n) {
        for (unsigned int i=0; i<wtimes_us.size(); ++i) {
            noop.nInitialThreads = n;
            team2.startCycle(noop, "quick", true);

            start = steady_clock::now();
            // Since true was passed above, all threads at the start of this
            // measurement should be in Wait
            team2.closeQueue();
            team2.wait();
            end = steady_clock::now();
            wtimes_us[i] = microseconds(end - start).count();
            EXPECT_TRUE(wtimes_us[i] > 0.0);
        }
        processWalltimes(wtimes_us, &mean_wtime_us, &std_wtime_us);
        std::cout << n << " thread/"
                  << 0 << " units of work enqueued/"
                  << "closeQueue Time/ wait=T\t\t" 
                  << mean_wtime_us << " +/- "
                  << std_wtime_us << " us\n";

        filename =   "CloseQueueTimings_"
                   + std::to_string(n)
                   + "_threads.dat";
        fptr.open(filename, std::ios::out);
        fptr << "# C++ steady_clock\n";
        fptr << "# Clock Resolution\t\t" << resolution_str << "\n";
        fptr << "N_threads,N_work,wtime_us\n";
        for (auto& wt_us : wtimes_us) {
            fptr << std::setprecision(15)
                 << n << ","
                 << 0 << ","
                 << wt_us << "\n";
        }
        fptr.close();
    }

    //***** PROBE INFORMATION ON Wait->Idle->Wait TRANSITION TIMES
    // This test is presently used to make certain that measuring startCycle and 
    // endCycle separately is occuring as I expect.  Specifically, if the test
    // setup is setup well, then adding the above time samples together should
    // approximately equal the values measured here.  This might not be true,
    // for instance, if I switch true to false below.
    for (unsigned int n=0; n<=T3::nThreadsPerTeam; ++n) {
        for (unsigned int i=0; i<wtimes_us.size(); ++i) {
            noop.nInitialThreads = n;

            start = steady_clock::now();
            team2.startCycle(noop, "quick", true);
            team2.closeQueue();
            team2.wait();
            end = steady_clock::now();
            wtimes_us[i] = microseconds(end - start).count();
            EXPECT_TRUE(wtimes_us[i] > 0.0);
        }
        processWalltimes(wtimes_us, &mean_wtime_us, &std_wtime_us);
        std::cout << n << " thread/"
                  << 0 << " units of work enqueued/"
                  << "single no-op Time/ wait=T\t\t" 
                  << mean_wtime_us << " +/- "
                  << std_wtime_us << " us\n";

        filename =   "NoWorkTimings_"
                   + std::to_string(n)
                   + "_threads.dat";
        fptr.open(filename, std::ios::out);
        fptr << "# C++ steady_clock\n";
        fptr << "# Clock Resolution\t\t" << resolution_str << "\n";
        fptr << "N_threads,N_work,wtime_us\n";
        for (auto& wt_us : wtimes_us) {
            fptr << std::setprecision(15)
                 << n << ","
                 << 0 << ","
                 << wt_us << "\n";
        }
        fptr.close();
    }

    //***** PROBE INFORMATION ON FULL CYCLE
    // Enqueue same amount of work for each case so that we can get an idea of
    // the overhead associated with the thread team
    for (unsigned int n=1; n<=T3::nThreadsPerTeam; ++n) {
        for (unsigned int i=0; i<wtimes_us.size(); ++i) {
            noop.nInitialThreads = n;
            start = steady_clock::now();
            team2.startCycle(noop, "quick", false);
            for (unsigned int j=0; j<N_WORK; ++j) {
                team2.enqueue(work, false);
            }
            team2.closeQueue();
            team2.wait();
            end = steady_clock::now();
            wtimes_us[i] = microseconds(end - start).count();
            EXPECT_TRUE(wtimes_us[i] > 0.0);
        }
        processWalltimes(wtimes_us, &mean_wtime_us, &std_wtime_us);
        std::cout << n      << " thread/"
                  << N_WORK << " units of work enqueued/"
                  << "single no-op cycle Time/ wait=F\t" 
                  << mean_wtime_us << " +/- "
                  << std_wtime_us << " us\n";

        filename =   "FullCycleTimings_"
                   + std::to_string(n)
                   + "_threads.dat";
        fptr.open(filename, std::ios::out);
        fptr << "# C++ steady_clock\n";
        fptr << "# Clock Resolution\t\t" << resolution_str << "\n";
        fptr << "N_threads,N_work,wtime_us\n";
        for (auto& wt_us : wtimes_us) {
            fptr << std::setprecision(15)
                 << n << ","
                 << N_WORK << ","
                 << wt_us << "\n";
        }
        fptr.close();
    }
}
#endif

}

