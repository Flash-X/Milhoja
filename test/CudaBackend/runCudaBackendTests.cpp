#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_RuntimeBackend.h>
#include <Milhoja_Logger.h>

namespace cudaTestConstants {
    unsigned int SLEEP_TIME_NS = 0;
};

int main(int argc, char* argv[]) {
    MPI_Comm      GLOBAL_COMM = MPI_COMM_WORLD;
    int           LEAD_RANK   = 0;

    // This value cannot be changed without breaking tests.
    constexpr int           N_STREAMS = 3;
    constexpr std::size_t   N_BYTES_IN_MEMORY_POOLS = 8;

    ::testing::InitGoogleTest(&argc, argv);

    if (argc != 2) {
        std::cerr << "\nOne and only one non-googletest argument please!\n\n";
        return 1;
    }
    cudaTestConstants::SLEEP_TIME_NS = std::stoi(std::string(argv[1]));

    MPI_Init(&argc, &argv);

    int     exitCode = 1;
    try {
        // Instantiate up front so that the acquisition of stream resources is not
        // included in the timing of the first test.
        milhoja::Logger::initialize("CudaBackend.log", GLOBAL_COMM, LEAD_RANK);
        milhoja::RuntimeBackend::initialize(N_STREAMS, N_BYTES_IN_MEMORY_POOLS);

        exitCode = RUN_ALL_TESTS();

        milhoja::RuntimeBackend::instance().finalize();
        milhoja::Logger::instance().finalize();
    } catch(const std::exception& e) {
        std::cerr << "FAILURE - CudaBackendTests::main - " << e.what() << std::endl;
        exitCode = 111;
    } catch(...) {
        std::cerr << "FAILURE - CudaBackendTests::main - Exception of unexpected type caught"
                  << std::endl;
        exitCode = 222;
    }

    MPI_Finalize();

    return exitCode;
}

