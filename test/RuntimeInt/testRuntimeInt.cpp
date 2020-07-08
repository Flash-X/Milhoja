#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <stdexcept>

#include "ThreadTeam.h"

#include "gtest/gtest.h"

using namespace orchestration;

namespace {

// No-op action routine that assume that give dataItem is an int
void noopActionRoutine_int(const int tId, void* dataItem) { }

// Action routine that assume that give dataItem is an int
// and that sleepsfor a random amount of time
void randomActionRoutine_int(const int tId, void* dataItem) {
    int  time = rand() % 100;
    std::this_thread::sleep_for(std::chrono::microseconds(time));
}

/**
 *   Define a test fixture
 */ 
class ThreadRuntimeInt : public testing::Test {
protected:
    RuntimeAction    noop_int;
    RuntimeAction    random_int;

    ThreadRuntimeInt(void) {
        noop_int.name = "noop_int";
        noop_int.nInitialThreads = 0;
        noop_int.teamType = ThreadTeamDataType::OTHER;
        noop_int.routine = noopActionRoutine_int;

        random_int.name = "Random Wait";
        random_int.nInitialThreads = 0;
        random_int.teamType = ThreadTeamDataType::OTHER;
        random_int.routine = randomActionRoutine_int;
    }

    ~ThreadRuntimeInt(void) { }
};

/**
 * Build up a runtime manually where the unit of work is a single unsigned int.
 * The test has three thread teams with both thread sub/pub and work sub/pub.
 * Only one execution cycle is run with the intent that the log file be manually
 * studied to confirm correctness.
 */
TEST_F(ThreadRuntimeInt, TestSingle_ManualCheck) {
    std::vector<int>   work = {-5, 4, -1, 0, -6, 25};

    // postGpu has enough threads to receive all of cpu and gpu threads
    ThreadTeam<int>   cpu_int(3,      1, "TestSingle_ManualCheck.log");
    ThreadTeam<int>   gpu_int(6,      2, "TestSingle_ManualCheck.log");
    ThreadTeam<int>   postGpu_int(10, 3, "TestSingle_ManualCheck.log");

    cpu_int.attachThreadReceiver(&postGpu_int);
    gpu_int.attachThreadReceiver(&postGpu_int);
    gpu_int.attachWorkReceiver(&postGpu_int);

    try {
        noop_int.nInitialThreads = 2;
        cpu_int.startTask(noop_int,     "Cpu");
        noop_int.nInitialThreads = 5;
        gpu_int.startTask(noop_int,     "Gpu");
        noop_int.nInitialThreads = 0;
        postGpu_int.startTask(noop_int, "postGpu");

        for (unsigned int i=0; i<work.size(); ++i) {
            cpu_int.enqueue(work[i], false);
            gpu_int.enqueue(work[i], true);
        }
        // gpu will call closeTask of postGpu when gpu transitions to Idle
        gpu_int.closeTask();
        cpu_int.closeTask();

        cpu_int.wait();
        gpu_int.wait();
        postGpu_int.wait();

        cpu_int.detachThreadReceiver();
        gpu_int.detachThreadReceiver();
        gpu_int.detachWorkReceiver();
    } catch (std::invalid_argument  e) {
        printf("\nINVALID ARGUMENT: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (std::logic_error  e) {
        printf("\nLOGIC ERROR: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (std::runtime_error  e) {
        printf("\nRUNTIME ERROR: %s\n\n", e.what());
        EXPECT_TRUE(false);
    } catch (...) {
        printf("\n??? ERROR: Unanticipated error\n\n");
        EXPECT_TRUE(false);
    }
}

/**
 * Build up a runtime manually where the unit of work is a single unsigned int.
 * The test has three thread teams with both thread sub/pub and work sub/pub.
 * This test is meant to run many no-op execution cycles using a realistic
 * topology to simply exercise the code by running the EFSM as quickly as
 * possible.  As no real work is being done, there is no means to check that the
 * runtime produced the correct result.  No exceptions means the test passed.
 */
TEST_F(ThreadRuntimeInt, TestMultipleFast) {
#ifdef DEBUG_RUNTIME
    unsigned int   N_ITERS    = 10;
#else
    unsigned int   N_ITERS    = 100;
#endif
    unsigned int   MAX_N_WORK = 1000;

    std::vector<int>   work(MAX_N_WORK);
    for (unsigned int i=0; i<MAX_N_WORK; ++i) {
        work[i] = i;
    }

    // postGpu has enough threads to receive all of cpu and gpu threads
    ThreadTeam<int>   cpu_int(3,      1, "TestMultipleFast.log");
    ThreadTeam<int>   gpu_int(6,      2, "TestMultipleFast.log");
    ThreadTeam<int>   postGpu_int(10, 3, "TestMultipleFast.log");

    for (unsigned int i=0; i<N_ITERS; ++i) {
        cpu_int.attachThreadReceiver(&postGpu_int);
        gpu_int.attachThreadReceiver(&postGpu_int);
        gpu_int.attachWorkReceiver(&postGpu_int);

        try {
            noop_int.nInitialThreads = 2;
            cpu_int.startTask(noop_int,     "Cpu");
            noop_int.nInitialThreads = 5;
            gpu_int.startTask(noop_int,     "Gpu");
            noop_int.nInitialThreads = 0;
            postGpu_int.startTask(noop_int, "postGpu");

            for (unsigned int i=0; i<work.size(); ++i) {
                cpu_int.enqueue(work[i], false);
                gpu_int.enqueue(work[i], true);
            }
            gpu_int.closeTask();
            cpu_int.closeTask();

            cpu_int.wait();
            gpu_int.wait();
            postGpu_int.wait();
        } catch (std::invalid_argument  e) {
            printf("\nINVALID ARGUMENT: %s\n\n", e.what());
            EXPECT_TRUE(false);
        } catch (std::logic_error  e) {
            printf("\nLOGIC ERROR: %s\n\n", e.what());
            EXPECT_TRUE(false);
        } catch (std::runtime_error  e) {
            printf("\nRUNTIME ERROR: %s\n\n", e.what());
            EXPECT_TRUE(false);
        } catch (...) {
            printf("\n??? ERROR: Unanticipated error\n\n");
            EXPECT_TRUE(false);
        }

        cpu_int.detachThreadReceiver();
        gpu_int.detachThreadReceiver();
        gpu_int.detachWorkReceiver();
    }
}

/**
 * Build up a runtime manually where the unit of work is a single unsigned int.
 * The test has three thread teams with both thread sub/pub and work sub/pub.
 * This test is meant to run many execution cycles using a realistic topology to
 * simply exercise the code.  In this case, each task call sleeps for a random
 * amount of time so that hopefully we are exercising the runtime with many
 * different execution sequences.  As no real work is being done, there is no
 * means to check that the runtime produced the correct result.  No exceptions
 * means the test passed.
 */
TEST_F(ThreadRuntimeInt, TestMultipleRandomWait) {
    srand(1000);

#ifdef DEBUG_RUNTIME
    unsigned int   N_ITERS    = 10;
#else
    unsigned int   N_ITERS    = 100;
#endif
    unsigned int   MAX_N_WORK = 1000;

    std::vector<int>   work(MAX_N_WORK);
    for (unsigned int i=0; i<MAX_N_WORK; ++i) {
        work[i] = i;
    }

    // postGpu has enough threads to receive all of cpu and gpu threads
    ThreadTeam<int>   cpu_int(3,      1, "TestMultipleRandomWait.log");
    ThreadTeam<int>   gpu_int(6,      2, "TestMultipleRandomWait.log");
    ThreadTeam<int>   postGpu_int(10, 3, "TestMultipleRandomWait.log");

    cpu_int.attachThreadReceiver(&postGpu_int);
    gpu_int.attachThreadReceiver(&postGpu_int);
    gpu_int.attachWorkReceiver(&postGpu_int);

    for (unsigned int i=0; i<N_ITERS; ++i) {
        try {
            random_int.nInitialThreads = 2;
            cpu_int.startTask(random_int,     "Cpu");
            random_int.nInitialThreads = 5;
            gpu_int.startTask(random_int,     "Gpu");
            random_int.nInitialThreads = 0;
            postGpu_int.startTask(random_int, "postGpu");

            for (unsigned int i=0; i<work.size(); ++i) {
                cpu_int.enqueue(work[i], false);
                gpu_int.enqueue(work[i], true);
            }
            gpu_int.closeTask();
            cpu_int.closeTask();

            cpu_int.wait();
            gpu_int.wait();
            postGpu_int.wait();
        } catch (std::invalid_argument  e) {
            printf("\nINVALID ARGUMENT: %s\n\n", e.what());
            EXPECT_TRUE(false);
        } catch (std::logic_error  e) {
            printf("\nLOGIC ERROR: %s\n\n", e.what());
            EXPECT_TRUE(false);
        } catch (std::runtime_error  e) {
            printf("\nRUNTIME ERROR: %s\n\n", e.what());
            EXPECT_TRUE(false);
        } catch (...) {
            printf("\n??? ERROR: Unanticipated error\n\n");
            EXPECT_TRUE(false);
        }
    }

    cpu_int.detachThreadReceiver();
    gpu_int.detachThreadReceiver();
    gpu_int.detachWorkReceiver();
}

}

