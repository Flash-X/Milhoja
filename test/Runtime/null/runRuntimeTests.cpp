#include <gtest/gtest.h>

#include <mpi.h>

#include <Milhoja_Logger.h>

int main(int argc, char* argv[]) {
    MPI_Comm   GLOBAL_COMM = MPI_COMM_WORLD;
    int        LEAD_RANK   = 0;

    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(&argc, &argv);

    int     exitCode = 1;
    try {
        milhoja::Logger::initialize("RuntimeTest.log",
                                    GLOBAL_COMM, LEAD_RANK);

        exitCode = RUN_ALL_TESTS();

        milhoja::Logger::instance().finalize();
    } catch(const std::exception& e) {
        std::cerr << "FAILURE - Runtime/null - " << e.what() << std::endl;
        exitCode = 111;
    } catch(...) {
        std::cerr << "FAILURE - Runtime::null - Exception of unexpected type caught"
                  << std::endl;
        exitCode = 222;
    }

    MPI_Finalize();

    return exitCode;
}

