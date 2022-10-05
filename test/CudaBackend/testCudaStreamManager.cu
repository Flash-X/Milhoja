#include <gtest/gtest.h>

#include <Milhoja.h>
#include <Milhoja_Logger.h>
#include <Milhoja_RuntimeBackend.h>

#include "cudaTestConstants.h"

#ifndef MILHOJA_CUDA_RUNTIME_BACKEND
#error "No sense in running this test if CUDA backend not chosen"
#endif

#ifndef MILHOJA_OPENACC_OFFLOADING
#error "Please enable offloading with OpenACC"
#endif

__global__ void kernel(const std::size_t N, double* f, const unsigned int sleepTime_ns) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        f[i] *= 2.2;
        __nanosleep(sleepTime_ns);
    }
}

namespace {

TEST(TestCudaStreamManager, TestManager) {
    using namespace milhoja;

    Logger::instance().log("[googletest] Start TestManager test");

    RuntimeBackend&   bknd = RuntimeBackend::instance();
    int  maxNumStreams = bknd.maxNumberStreams();
    ASSERT_EQ(3, maxNumStreams);
    ASSERT_EQ(maxNumStreams, bknd.numberFreeStreams());

    // Confirm that streams are null streams by default
    Stream   streams[maxNumStreams];
    for (std::size_t i=0; i<maxNumStreams; ++i) {
        ASSERT_EQ(NULL_ACC_ASYNC_QUEUE, streams[i].accAsyncQueue);
        ASSERT_EQ(nullptr,              streams[i].cudaStream);
    }

    // Check out all available streams and confirm that they are valid streams
    streams[0] = bknd.requestStream(true); 
    streams[1] = bknd.requestStream(true); 
    streams[2] = bknd.requestStream(true); 
    ASSERT_EQ(0, bknd.numberFreeStreams());
    EXPECT_NE(NULL_ACC_ASYNC_QUEUE, streams[0].accAsyncQueue);
    EXPECT_NE(nullptr,              streams[0].cudaStream);
    EXPECT_NE(NULL_ACC_ASYNC_QUEUE, streams[1].accAsyncQueue);
    EXPECT_NE(nullptr,              streams[1].cudaStream);
    EXPECT_NE(NULL_ACC_ASYNC_QUEUE, streams[2].accAsyncQueue);
    EXPECT_NE(nullptr,              streams[2].cudaStream);

    // TODO: Get start time, call request, wait for a given amount of time,
    //       have another thread
    //       free a stream, and them confirm that time blocked is >= given time.
//    Stream    blockedStream = bknd.requestStream(true);
    
    // If there are no free streams and we don't want to be blocked, then the
    // returned stream should be a null stream.
    Stream    nullStream = bknd.requestStream(false);
    EXPECT_EQ(NULL_ACC_ASYNC_QUEUE, nullStream.accAsyncQueue);
    EXPECT_EQ(nullptr,              nullStream.cudaStream);

    // Confirm that releasing a stream nullifies my stream.
    bknd.releaseStream(streams[0]);
    ASSERT_EQ(1, bknd.numberFreeStreams());
    EXPECT_EQ(NULL_ACC_ASYNC_QUEUE, streams[0].accAsyncQueue);
    EXPECT_EQ(nullptr,              streams[0].cudaStream);

    streams[0] = bknd.requestStream(true); 
    ASSERT_EQ(0, bknd.numberFreeStreams());
    EXPECT_NE(NULL_ACC_ASYNC_QUEUE, streams[0].accAsyncQueue);
    EXPECT_NE(nullptr,              streams[0].cudaStream);

    bknd.releaseStream(streams[0]); 
    bknd.releaseStream(streams[2]); 
    bknd.releaseStream(streams[1]); 
    ASSERT_EQ(maxNumStreams, bknd.numberFreeStreams());
    EXPECT_EQ(NULL_ACC_ASYNC_QUEUE, streams[0].accAsyncQueue);
    EXPECT_EQ(nullptr,              streams[0].cudaStream);
    EXPECT_EQ(NULL_ACC_ASYNC_QUEUE, streams[1].accAsyncQueue);
    EXPECT_EQ(nullptr,              streams[1].cudaStream);
    EXPECT_EQ(NULL_ACC_ASYNC_QUEUE, streams[2].accAsyncQueue);
    EXPECT_EQ(nullptr,              streams[2].cudaStream);

    // TEST ERRORS ASSOCIATED WITH RELEASING STREAMS
    // Run the tests when there is still one stream to return
    // so that we cannot mistake the desired error with the error of releasing
    // more streams than the manager owns.
    streams[0] = bknd.requestStream(true);
    Stream   goodStream;
    Stream   goodStream2;
    goodStream.accAsyncQueue  = streams[0].accAsyncQueue;
    goodStream.cudaStream     = streams[0].cudaStream;
    goodStream2.accAsyncQueue = goodStream.accAsyncQueue;
    goodStream2.cudaStream    = goodStream.cudaStream;

    // Cannot release a stream with a null OpenACC async queue ID
    Stream   badStream;
    try {
        badStream.accAsyncQueue = NULL_ACC_ASYNC_QUEUE;
        badStream.cudaStream    = goodStream.cudaStream;
        bknd.releaseStream(badStream);
        EXPECT_TRUE(false);
    } catch(const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    // The stream pointer cannot be null
    try {
        badStream.accAsyncQueue = goodStream.accAsyncQueue;
        badStream.cudaStream    = nullptr;
        bknd.releaseStream(badStream);
        EXPECT_TRUE(false);
    } catch(const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    // Return the final stream and try pushing a "valid" stream (i.e. ID and
    // object not null) when the manager has all its streams accounted for.
    bknd.releaseStream(goodStream);
    ASSERT_EQ(maxNumStreams, bknd.numberFreeStreams());
    try {
        bknd.releaseStream(goodStream2);
        EXPECT_TRUE(false);
    } catch (const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    Logger::instance().log("[googletest] End TestManager test");
}

/**
 *  It is intended that this test be run with nvprof and that the output of this
 *  be manually inspected to confirm that streaming was used and functioned as
 *  expected.
 */
TEST(TestCudaStreamManager, TestStreams) {
    using namespace milhoja;

    Logger::instance().log("[googletest] Start TestStreams test");

    // We will send one packet of equal size per stream and break each packet up
    // into smaller equal-sized chunks for computation with GPU.
    constexpr  std::size_t  N_DATA_PER_PACKET = 1024;

    RuntimeBackend&   bknd = RuntimeBackend::instance();
    int  maxNumStreams = bknd.maxNumberStreams();
    ASSERT_EQ(maxNumStreams, bknd.numberFreeStreams());

    int  nPackets = maxNumStreams;
    int  nData = nPackets * N_DATA_PER_PACKET;

    Stream   streams[maxNumStreams];
    for (unsigned int i=0; i<maxNumStreams; ++i) {
        streams[i] = bknd.requestStream(true); 
    }
    ASSERT_EQ(0, bknd.numberFreeStreams());

    double*        data_p = nullptr;
    double*        data_d = nullptr;
    unsigned int   nBytes = nData * sizeof(double);
    cudaError_t    cErr = cudaMallocHost(&data_p, nBytes);
    ASSERT_EQ(cudaSuccess, cErr);
    cErr = cudaMalloc(&data_d, nBytes);
    ASSERT_EQ(cudaSuccess, cErr);

    for (std::size_t i=0; i<nData; ++i) {
        data_p[i] = 1.1;
    }

    for (unsigned int i=0; i<maxNumStreams; ++i) {
        cudaStream_t  myStream = streams[i].cudaStream;
        unsigned int  offset = i * N_DATA_PER_PACKET;
        cErr = cudaMemcpyAsync(data_d + offset,
                               data_p + offset,
                               N_DATA_PER_PACKET * sizeof(double),
                               cudaMemcpyHostToDevice,
                               myStream);
        ASSERT_EQ(cudaSuccess, cErr);
        kernel<<<N_DATA_PER_PACKET,128,0,myStream>>>(N_DATA_PER_PACKET,
                                                     data_d + offset,
                                                     cudaTestConstants::SLEEP_TIME_NS);
        cErr = cudaMemcpyAsync(data_p + offset,
                               data_d + offset,
                               N_DATA_PER_PACKET * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               myStream);
        ASSERT_EQ(cudaSuccess, cErr);
    }
    cErr = cudaDeviceSynchronize();
    ASSERT_EQ(cudaSuccess, cErr);

    for (std::size_t i=0; i<nData; ++i) {
        EXPECT_NEAR(2.42, data_p[i], 1.0e-15);
    }

    cudaFree(data_d);
    cudaFreeHost(data_p);
    data_d = nullptr;
    data_p = nullptr;

    for (unsigned int i=0; i<maxNumStreams; ++i ) {
        bknd.releaseStream(streams[i]); 
    }
    ASSERT_EQ(maxNumStreams, bknd.numberFreeStreams());

    Logger::instance().log("[googletest] End TestStreams test");
}

// TODO: Put in a test that exercises the manager using many different threads
//       and in blocking mode.  The threads should request a stream and then
//       wait for a random amount of time before releaseing the stream.
//       This could emulate the structure of a simulation where the manager
//       should have all streams free, get a burst of activity with the burst
//       terminating with all streams free again.  We could them loop over this
//       so that there are many bursts of activity.
}

