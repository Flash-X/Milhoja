#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <stdexcept>

#include "ThreadTeam.h"

#include "gtest/gtest.h"

namespace {

// No-op set of task routines
void cpuNoop(const unsigned int tId,
             const std::string& name,
             const unsigned int work) { }

void gpuNoop(const unsigned int tId,
             const std::string& name,
             const unsigned int work) { }

void postGpuNoop(const unsigned int tId,
                 const std::string& name,
                 const unsigned int work) { }

// Set of task routines that sleep for a random amount of time
void cpuRandom(const unsigned int tId,
               const std::string& name,
               const unsigned int work) {
    int  time = rand() % 100;
    std::this_thread::sleep_for(std::chrono::microseconds(time));
}

void gpuRandom(const unsigned int tId,
               const std::string& name,
               const unsigned int work) {
    int  time = rand() % 100;
    std::this_thread::sleep_for(std::chrono::microseconds(time));
}

void postGpuRandom(const unsigned int tId,
                   const std::string& name,
                   const unsigned int work) {
    int  time = rand() % 100;
    std::this_thread::sleep_for(std::chrono::microseconds(time));
}

/**
 * Build up a runtime manually where the unit of work is a single unsigned int.
 * The test has three thread teams with both thread sub/pub and work sub/pub.
 * Only one execution cycle is run with the intent that the log file be manually
 * studied to confirm correctness.
 */
TEST(ThreadRuntimeInt, TestSingle_ManualCheck) {
    std::vector<unsigned int>   work = {5, 4, 1, 0, 6, 25};

    // postGpu has enough threads to receive all of cpu and gpu threads
    ThreadTeam   cpu(3,      1, "TestSingle_ManualCheck.log");
    ThreadTeam   gpu(6,      2, "TestSingle_ManualCheck.log");
    ThreadTeam   postGpu(10, 3, "TestSingle_ManualCheck.log");

    cpu.attachThreadReceiver(&postGpu);
    gpu.attachThreadReceiver(&postGpu);
    gpu.attachWorkReceiver(&postGpu);

    try {
        cpu.startTask(cpuNoop,         2, "Cpu",     "cpuNoop");
        gpu.startTask(gpuNoop,         5, "Gpu",     "gpuNoop");
        postGpu.startTask(postGpuNoop, 0, "postGpu", "postGpuNoop");

        for (auto w: work) {
            cpu.enqueue(w);
            gpu.enqueue(w);
        }
        // gpu will call closeTask of postGpu when gpu transitions to Idle
        gpu.closeTask();
        cpu.closeTask();

        cpu.wait();
        gpu.wait();
        postGpu.wait();

        cpu.detachThreadReceiver();
        gpu.detachThreadReceiver();
        gpu.detachWorkReceiver();
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
TEST(ThreadRuntimeInt, TestMultipleFast) {
#ifdef VERBOSE
    unsigned int   N_ITERS    = 10;
#else
    unsigned int   N_ITERS    = 100;
#endif
    unsigned int   MAX_N_WORK = 1000;

    std::vector<unsigned int>   work(MAX_N_WORK);
    for (unsigned int i=0; i<MAX_N_WORK; ++i) {
        work[i] = i;
    }

    // postGpu has enough threads to receive all of cpu and gpu threads
    ThreadTeam   cpu(3,      1, "TestMultipleFast.log");
    ThreadTeam   gpu(6,      2, "TestMultipleFast.log");
    ThreadTeam   postGpu(10, 3, "TestMultipleFast.log");

    for (unsigned int i=0; i<N_ITERS; ++i) {
        cpu.attachThreadReceiver(&postGpu);
        gpu.attachThreadReceiver(&postGpu);
        gpu.attachWorkReceiver(&postGpu);

        try {
            cpu.startTask(cpuNoop,         2, "Cpu",     "cpuNoop");
            gpu.startTask(gpuNoop,         5, "Gpu",     "gpuNoop");
            postGpu.startTask(postGpuNoop, 0, "postGpu", "postGpuNoop");

            for (auto w: work) {
                cpu.enqueue(w);
                gpu.enqueue(w);
            }
            gpu.closeTask();
            cpu.closeTask();

            cpu.wait();
            gpu.wait();
            postGpu.wait();
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

        cpu.detachThreadReceiver();
        gpu.detachThreadReceiver();
        gpu.detachWorkReceiver();
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
TEST(ThreadRuntimeInt, TestMultipleRandomWait) {
    srand(1000);

#ifdef VERBOSE
    unsigned int   N_ITERS    = 10;
#else
    unsigned int   N_ITERS    = 100;
#endif
    unsigned int   MAX_N_WORK = 1000;

    std::vector<unsigned int>   work(MAX_N_WORK);
    for (unsigned int i=0; i<MAX_N_WORK; ++i) {
        work[i] = i;
    }

    // postGpu has enough threads to receive all of cpu and gpu threads
    ThreadTeam   cpu(3,      1, "TestMultipleRandomWait.log");
    ThreadTeam   gpu(6,      2, "TestMultipleRandomWait.log");
    ThreadTeam   postGpu(10, 3, "TestMultipleRandomWait.log");

    cpu.attachThreadReceiver(&postGpu);
    gpu.attachThreadReceiver(&postGpu);
    gpu.attachWorkReceiver(&postGpu);

    for (unsigned int i=0; i<N_ITERS; ++i) {
        try {
            cpu.startTask(cpuRandom,         2, "Cpu",     "cpuRandom");
            gpu.startTask(gpuRandom,         5, "Gpu",     "gpuRandom");
            postGpu.startTask(postGpuRandom, 0, "postGpu", "postGpuRandom");

            for (auto w: work) {
                cpu.enqueue(w);
                gpu.enqueue(w);
            }
            gpu.closeTask();
            cpu.closeTask();

            cpu.wait();
            gpu.wait();
            postGpu.wait();
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

    cpu.detachThreadReceiver();
    gpu.detachThreadReceiver();
    gpu.detachWorkReceiver();
}

}

