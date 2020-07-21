#include "cudaTestConstants.h"
#include "CudaStreamManager.h"

#include "gtest/gtest.h"

__global__ void kernel(const std::size_t N, double* f, const unsigned int nCycles) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        f[i] *= 2.2;

        // Put thread in busy wait loop so that we can correctly
        // detect overlaying of computation and communication and
        // therefore verify that streams are being used correctly.
        clock_t clock_offset = 0;
        clock_t start_clock = clock64();
        while (clock_offset < nCycles) {
            clock_offset = clock64() - start_clock;
        }
    }
}

namespace {

TEST(TestCudaStreamManager, TestManager) {
    CudaStreamManager&   sm = CudaStreamManager::instance();
    ASSERT_EQ(cudaTestConstants::N_STREAMS, sm.numberFreeStreams());

    // Confirm that streams are null streams by default
    CudaStream   streams[cudaTestConstants::N_STREAMS];
    for (std::size_t i=0; i<cudaTestConstants::N_STREAMS; ++i) {
        ASSERT_EQ(CudaStream::NULL_STREAM_ID, streams[i].id);
        ASSERT_EQ(nullptr,                    streams[i].object);
    }

    // Check out all available streams and confirm that they are valid streams
    streams[0] = sm.requestStream(true); 
    streams[1] = sm.requestStream(true); 
    streams[2] = sm.requestStream(true); 
    ASSERT_EQ(0, sm.numberFreeStreams());
    EXPECT_NE(CudaStream::NULL_STREAM_ID, streams[0].id);
    EXPECT_NE(nullptr,                    streams[0].object);
    EXPECT_NE(CudaStream::NULL_STREAM_ID, streams[1].id);
    EXPECT_NE(nullptr,                    streams[1].object);
    EXPECT_NE(CudaStream::NULL_STREAM_ID, streams[2].id);
    EXPECT_NE(nullptr,                    streams[2].object);

    // TODO: Get start time, call request, wait for a given amount of time,
    //       have another thread
    //       free a stream, and them confirm that time blocked is >= given time.
//    CudaStream    blockedStream = sm.requestStream(true);
    
    // If there are no free streams and we don't want to be blocked, then the
    // returned stream should be a null stream.
    CudaStream    nullStream = sm.requestStream(false);
    EXPECT_EQ(CudaStream::NULL_STREAM_ID, nullStream.id);
    EXPECT_EQ(nullptr,                    nullStream.object);

    // Confirm that releasing a stream nullifies my stream.
    sm.releaseStream(streams[0]);
    ASSERT_EQ(1, sm.numberFreeStreams());
    EXPECT_EQ(CudaStream::NULL_STREAM_ID, streams[0].id);
    EXPECT_EQ(nullptr,                    streams[0].object);

    streams[0] = sm.requestStream(true); 
    ASSERT_EQ(0, sm.numberFreeStreams());
    EXPECT_NE(CudaStream::NULL_STREAM_ID, streams[0].id);
    EXPECT_NE(nullptr,                    streams[0].object);

    // A double release should fail as the stream should not be free on the
    // second call.  Try the double call now so that we know that the error
    // that we would like to test is not mixed with the error of returning too
    // many streams.
    CudaStream  goodStream;
    goodStream.id      = streams[0].id;
    goodStream.object  = streams[0].object;

    // Create a counterfeit stream.  The ID doesn't match the stream pointer.
    // Use the ID of streams[1] as that stream will still be out at the time of
    // the test.
    CudaStream  badStream;
    badStream.id      = streams[1].id;
    badStream.object  = streams[0].object;

    // Double release should fail
    sm.releaseStream(streams[0]); 
    try {
        sm.releaseStream(goodStream); 
        EXPECT_TRUE(false);
    } catch(const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    // Confirm that counterfeit streams can't be released.
    try {
        sm.releaseStream(badStream); 
        EXPECT_TRUE(false);
    } catch(const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    sm.releaseStream(streams[2]); 
    sm.releaseStream(streams[1]); 
    ASSERT_EQ(cudaTestConstants::N_STREAMS, sm.numberFreeStreams());
    EXPECT_EQ(CudaStream::NULL_STREAM_ID, streams[0].id);
    EXPECT_EQ(nullptr,                    streams[0].object);
    EXPECT_EQ(CudaStream::NULL_STREAM_ID, streams[1].id);
    EXPECT_EQ(nullptr,                    streams[1].object);
    EXPECT_EQ(CudaStream::NULL_STREAM_ID, streams[2].id);
    EXPECT_EQ(nullptr,                    streams[2].object);

    // TEST ERRORS ASSOCIATED WITH RELEASING STREAMS
    // Run the tests when there is still one stream to return
    // so that we cannot mistake the desired error with the error of releasing
    // more streams than the manager owns.
    streams[0] = sm.requestStream(true);
    CudaStream   goodStream2;
    goodStream.id      = streams[0].id;
    goodStream.object  = streams[0].object;
    goodStream2.id     = goodStream.id;
    goodStream2.object = goodStream.object;

    // Cannot release a stream with a null ID
    try {
        badStream.id     = CudaStream::NULL_STREAM_ID;
        badStream.object = goodStream.object;
        sm.releaseStream(badStream);
        EXPECT_TRUE(false);
    } catch(const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    // The stream pointer cannot be null
    try {
        badStream.id     = goodStream.id;
        badStream.object = nullptr;
        sm.releaseStream(badStream);
        EXPECT_TRUE(false);
    } catch(const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    // The stream ID must be valid
    try {
        badStream.id     = cudaTestConstants::N_STREAMS + 1;
        badStream.object = goodStream.object;
        sm.releaseStream(badStream);
        EXPECT_TRUE(false);
    } catch(const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }

    // Return the final stream and try pushing a "valid" stream (i.e. ID and
    // object not null) when the manager has all its streams accounted for.
    sm.releaseStream(goodStream);
    ASSERT_EQ(cudaTestConstants::N_STREAMS, sm.numberFreeStreams());
    try {
        sm.releaseStream(goodStream2);
        EXPECT_TRUE(false);
    } catch (const std::invalid_argument&) {
        EXPECT_TRUE(true);
    }
}

/**
 *  It is intended that this test be run with nvprof and that the output of this
 *  be manually inspected to confirm that streaming was used and functioned as
 *  expected.
 */
TEST(TestCudaStreamManager, TestStreams) {
    // We will send one packet of equal size per stream and break each packet up
    // into smaller equal-sized chunks for computation with GPU.
    constexpr  std::size_t  N_DATA_PER_PACKET = 1024;
    constexpr  std::size_t  N_PACKETS = cudaTestConstants::N_STREAMS;
    constexpr  std::size_t  N_DATA = N_PACKETS * N_DATA_PER_PACKET;

    CudaStreamManager&   sm = CudaStreamManager::instance();
    ASSERT_EQ(cudaTestConstants::N_STREAMS, sm.numberFreeStreams());

    CudaStream   streams[cudaTestConstants::N_STREAMS];
    for (unsigned int i=0; i<cudaTestConstants::N_STREAMS; ++i) {
        streams[i] = sm.requestStream(true); 
    }
    ASSERT_EQ(0, sm.numberFreeStreams());

    double*        data_p = nullptr;
    double*        data_d = nullptr;
    unsigned int   nBytes = N_DATA * sizeof(double);
    cudaError_t    cErr = cudaErrorInvalidValue;
    cErr = cudaMallocHost(&data_p, nBytes);
    ASSERT_EQ(cudaSuccess, cErr);
    cErr = cudaMalloc(&data_d, nBytes);
    ASSERT_EQ(cudaSuccess, cErr);

    for (std::size_t i=0; i<N_DATA; ++i) {
        data_p[i] = 1.1;
    }

    for (unsigned int i=0; i<cudaTestConstants::N_STREAMS; ++i) {
        cudaStream_t  myStream = *(streams[i].object);
        unsigned int  offset = i * N_DATA_PER_PACKET;
        cErr = cudaMemcpyAsync(data_d + offset,
                               data_p + offset,
                               N_DATA_PER_PACKET * sizeof(double),
                               cudaMemcpyHostToDevice,
                               myStream);
        ASSERT_EQ(cudaSuccess, cErr);
        kernel<<<N_DATA_PER_PACKET,128,0,myStream>>>(N_DATA_PER_PACKET,
                                                     data_d + offset,
                                                     cudaTestConstants::N_WAIT_CYCLES);
        cErr = cudaMemcpyAsync(data_p + offset,
                               data_d + offset,
                               N_DATA_PER_PACKET * sizeof(double),
                               cudaMemcpyDeviceToHost,
                               myStream);
        ASSERT_EQ(cudaSuccess, cErr);
    }
    cErr = cudaDeviceSynchronize();
    ASSERT_EQ(cudaSuccess, cErr);

    for (std::size_t i=0; i<N_DATA; ++i) {
        EXPECT_NEAR(2.42, data_p[i], 1.0e-15);
    }

    cudaFree(data_d);
    cudaFreeHost(data_p);
    data_d = nullptr;
    data_p = nullptr;

    for (unsigned int i=0; i<cudaTestConstants::N_STREAMS; ++i ) {
        sm.releaseStream(streams[i]); 
    }
    ASSERT_EQ(cudaTestConstants::N_STREAMS, sm.numberFreeStreams());
}

// TODO: Put in a test that exercises the manager using many different threads
//       and in blocking mode.  The threads should request a stream and then
//       wait for a random amount of time before releaseing the stream.
//       This could emulate the structure of a simulation where the manager
//       should have all streams free, get a burst of activity with the burst
//       terminating with all streams free again.  We could them loop over this
//       so that there are many bursts of activity.
}
