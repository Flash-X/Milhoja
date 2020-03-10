#include <AMReX.H>

#include <gtest/gtest.h>

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    // We need to create our own main for the testsuite since we can only call
    // MPI_Init/MPI_Finalize once per testsuite execution.
    amrex::Initialize(MPI_COMM_WORLD);
    int  errorCode = RUN_ALL_TESTS();
    amrex::Finalize();

    return errorCode;
}

