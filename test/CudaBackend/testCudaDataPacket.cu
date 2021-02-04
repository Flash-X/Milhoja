#include <AMReX_Dim3.H>
#include <AMReX_Array4.H>
#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>

#include "Tile.h"
#include "Grid.h"
#include "CudaDataPacket.h"
#include "CudaStreamManager.h"

#include "Flash.h"
#include "Flash_par.h"
#include "constants.h"

#include "gpuKernel.h"

#include "gtest/gtest.h"

namespace {

/**
 * Define a test fixture
 */
class TestCudaDataPacket : public testing::Test {
protected:
    static constexpr unsigned int   LEVEL = 0;

    static void setInitialConditions_block(const int tId, void* dataItem) {
        Tile*  tileDesc = static_cast<Tile*>(dataItem);

        Grid&    grid = Grid::instance();
        amrex::MultiFab&    unk = grid.unk();
        amrex::FArrayBox&   fab = unk[tileDesc->gridIndex()];

        amrex::Array4<amrex::Real> const&   f = fab.array();
    
        // Fill in the GC data as well as we aren't doing a GC fill in any
        // of these tests
        const amrex::Dim3 loGC = tileDesc->loGC();
        const amrex::Dim3 hiGC = tileDesc->hiGC();
        for     (int j = loGC.y; j <= hiGC.y; ++j) {
            for (int i = loGC.x; i <= hiGC.x; ++i) {
                f(i, j, loGC.z, DENS_VAR_C) = 2.2;
                f(i, j, loGC.z, ENER_VAR_C) = 2.2;
//                f(i, j, loGC.z, DENS_VAR_C) = i;
//                f(i, j, loGC.z, ENER_VAR_C) = 2.0 * j;
            }
        }
    }

    TestCudaDataPacket(void) {
        Grid&    grid = Grid::instance();
        grid.initDomain(rp_Grid::X_MIN, rp_Grid::X_MAX,
                        rp_Grid::Y_MIN, rp_Grid::Y_MAX,
                        rp_Grid::Z_MIN, rp_Grid::Z_MAX,
                        rp_Grid::N_BLOCKS_X,
                        rp_Grid::N_BLOCKS_Y,
                        rp_Grid::N_BLOCKS_Z,
                        NUNKVAR, 
                        TestCudaDataPacket::setInitialConditions_block);
   }

    ~TestCudaDataPacket(void) {
        Grid::instance().destroyDomain();
    }
};

TEST_F(TestCudaDataPacket, TestNullPacket) {
    CudaDataPacket    packet;
    EXPECT_TRUE(packet.isNull());
    EXPECT_EQ(0, packet.nDataItems());

    // Nullifying a null packet is acceptable
    packet.nullify();
    EXPECT_TRUE(packet.isNull());

    // Any attempt to access a data item should fail since empty
    try {
        Tile&  dataItem = packet[0];
        EXPECT_TRUE(false);
    } catch (std::out_of_range&) {
        EXPECT_TRUE(true);
    }

    try {
        const Tile&  dataItem = packet[0];
        EXPECT_TRUE(false);
    } catch (std::out_of_range&) {
        EXPECT_TRUE(true);
    }

    // No sense to prepare an empty packet for transfer
    try {
        packet.prepareForTransfer(0);
        EXPECT_TRUE(false);
    } catch (std::logic_error&) {
        EXPECT_TRUE(true);
    }

    // No sense to move data for an empty packet
    try {
        packet.moveDataFromPinnedToSource();
        EXPECT_TRUE(false);
    } catch (std::logic_error&) {
        EXPECT_TRUE(true);
    }
}

TEST_F(TestCudaDataPacket, TestNullifyPacket) {
    amrex::MultiFab&   unk = Grid::instance().unk();

    CudaDataPacket    packet;
    EXPECT_TRUE(packet.isNull());

    Tile dataItem;
    for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
        packet.addDataItem( std::move(Tile(itor, LEVEL)) );
    }
    EXPECT_FALSE(packet.isNull());
    packet.prepareForTransfer(0);

    EXPECT_FALSE(packet.isNull());
    packet.nullify();
    EXPECT_TRUE(packet.isNull());
}

TEST_F(TestCudaDataPacket, TestNDataItems) {
    constexpr unsigned int   N_BLOCKS =   rp_Grid::N_BLOCKS_X
                                        * rp_Grid::N_BLOCKS_Y
                                        * rp_Grid::N_BLOCKS_Z; 

    Tile dataItem;
    amrex::MultiFab&   unk = Grid::instance().unk();

    CudaDataPacket    packet;

    std::size_t   nAdded = 0;
    EXPECT_EQ(nAdded, packet.nDataItems());
    for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
        packet.addDataItem( Tile(itor, LEVEL) );
        EXPECT_EQ(++nAdded, packet.nDataItems());
    }
    EXPECT_EQ(N_BLOCKS, packet.nDataItems());

    packet.nullify();
    EXPECT_EQ(0, packet.nDataItems());
}

TEST_F(TestCudaDataPacket, TestAddDataItem) {
    constexpr unsigned int   N_BLOCKS =   rp_Grid::N_BLOCKS_X
                                        * rp_Grid::N_BLOCKS_Y
                                        * rp_Grid::N_BLOCKS_Z; 

    amrex::MultiFab&   unk = Grid::instance().unk();

    CudaDataPacket    packet1;
    CudaDataPacket    packet2;

    Tile dataItem;
    EXPECT_TRUE(dataItem.isNull());

    std::size_t   idx = 0;
    for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
        dataItem = Tile(itor, LEVEL);

        // The Tile should be left intact with the implicit copy
        // but nullified after the explicit move.
        EXPECT_FALSE(dataItem.isNull());
        packet1.addDataItem(           dataItem  );
        EXPECT_FALSE(dataItem.isNull());
    
        // Since addDataItem was a copy, we can confirm that packet1 
        // has the correct data
        const Tile& tmpItem1 = packet1[idx];
        EXPECT_EQ(dataItem.gridIndex(), tmpItem1.gridIndex());
        EXPECT_EQ(dataItem.level(),     tmpItem1.level());

        packet2.addDataItem( std::move(dataItem) );
        EXPECT_TRUE(dataItem.isNull());

        // move => check 2 against 1
        const Tile& tmpItem2 = packet2[idx];
        EXPECT_EQ(tmpItem1.gridIndex(), tmpItem2.gridIndex());
        EXPECT_EQ(tmpItem1.level(),     tmpItem2.level());

        ++idx;
    }
    EXPECT_EQ(N_BLOCKS, packet1.nDataItems());
    EXPECT_EQ(packet1.nDataItems(), packet2.nDataItems());

    // Confirm that two packets are still identical after finishing
    // the process of adding all dataItems.
    for (std::size_t i=0; i<packet1.nDataItems(); ++i) {
        const Tile& tmpItem1 = packet1[i];
        const Tile& tmpItem2 = packet2[i];
        EXPECT_EQ(tmpItem1.gridIndex(), tmpItem2.gridIndex());
        EXPECT_EQ(tmpItem1.level(),     tmpItem2.level());
    }
}

TEST_F(TestCudaDataPacket, TestPrepareForTransfer) {
    // Add all tiles to packet
    unsigned int n = 0;
    CudaDataPacket    packet;
    amrex::MultiFab&   unk = Grid::instance().unk();
    for (amrex::MFIter  itor(unk); itor.isValid(); ++itor) {
        // Put tile directly in packet using move
        packet.addDataItem( Tile(itor, LEVEL) );

        // Confirm my understanding of how Array4 works
        Tile const&  dataItem = packet[n];
        Tile         tileDesc = Tile(itor, LEVEL);
        amrex::FArrayBox&   fab = unk[tileDesc.gridIndex()];
        amrex::Array4<amrex::Real> const&   f = fab.array();
    
        EXPECT_EQ(fab.dataPtr(), f.dataPtr());
        EXPECT_EQ(f.dataPtr(),   tileDesc.CC_h_);
        EXPECT_EQ(f.dataPtr(),   dataItem.CC_h_);
        EXPECT_EQ(NUNKVAR, f.nComp());

        const amrex::Dim3 lbound = amrex::lbound(f);
        const amrex::Dim3 ubound = amrex::ubound(f);
        const amrex::Dim3 loGC   = tileDesc.loGC();
        const amrex::Dim3 hiGC   = tileDesc.hiGC();
        EXPECT_EQ(lbound.x, loGC.x);
        EXPECT_EQ(lbound.y, loGC.y);
        EXPECT_EQ(lbound.z, loGC.z);
        EXPECT_EQ(ubound.x, hiGC.x);
        EXPECT_EQ(ubound.y, hiGC.y);
        EXPECT_EQ(ubound.z, hiGC.z);

        ++n;
    }

    // Confirm correct data in source memory using only contents of data items
    // in the packet.
    for (std::size_t n=0; n<packet.nDataItems(); ++n) {
        const Tile&   dataItem = packet[n];

        ASSERT_TRUE(dataItem.CC_h_  != nullptr);
        EXPECT_TRUE(dataItem.CC1_p_ == nullptr);
        EXPECT_TRUE(dataItem.CC2_p_ == nullptr);
        EXPECT_TRUE(dataItem.CC1_d_ == nullptr);
        EXPECT_TRUE(dataItem.CC2_d_ == nullptr);

        const amrex::Dim3 loGC  = dataItem.loGC();
        const amrex::Dim3 hiGC  = dataItem.hiGC();
        const amrex::Dim3 begin = loGC;
        const amrex::Dim3 end{hiGC.x+1, hiGC.y+1, hiGC.z+1};

        amrex::Array4<amrex::Real> f(dataItem.CC_h_, begin, end, NUNKVAR);
        const amrex::Dim3 lbound = amrex::lbound(f);
        const amrex::Dim3 ubound = amrex::ubound(f);
        EXPECT_EQ(lbound.x, loGC.x);
        EXPECT_EQ(lbound.y, loGC.y);
        EXPECT_EQ(lbound.z, loGC.z);
        EXPECT_EQ(ubound.x, hiGC.x);
        EXPECT_EQ(ubound.y, hiGC.y);
        EXPECT_EQ(ubound.z, hiGC.z);

        for     (int j = loGC.y; j <= hiGC.y; ++j) {
            for (int i = loGC.x; i <= hiGC.x; ++i) {
                EXPECT_EQ(2.2, f(i, j, loGC.z, DENS_VAR_C));
                EXPECT_EQ(2.2, f(i, j, loGC.z, ENER_VAR_C));
//                EXPECT_EQ(i,       f(i, j, loGC.z, DENS_VAR_C));
//                EXPECT_EQ(2.0 * j, f(i, j, loGC.z, ENER_VAR_C));
            }
        }
    }

    // FIXME: nDataPerTile is presently fixed to a block
    gpuKernel::copyIn    copyInData;
    copyInData.nDataPerTile =   (NXB + 2 * NGUARD * K1D)
                              * (NYB + 2 * NGUARD * K2D)
                              * (NZB + 2 * NGUARD * K3D)
                              * NUNKVAR;
    copyInData.coefficient = 1.1;
    std::size_t   nBytesCopyIn = sizeof(gpuKernel::copyIn);

    // Ask packet to pack data for transfer
    // Confirm that data pointers set and that data is 
    // correct in pinned memory
    packet.prepareForTransfer(nBytesCopyIn);
    // TODO: prepareForTransfer() could just take a void pointer
    //       to the copyIn struct and do the copy for us.
    memcpy((void *)packet.copyIn_p_,
           (void *)(&copyInData),
           nBytesCopyIn);

    for (std::size_t n=0; n<packet.nDataItems(); ++n) {
        const Tile&   dataItem = packet[n];

        EXPECT_TRUE(dataItem.CC_h_  != nullptr);
        ASSERT_TRUE(dataItem.CC1_p_ != nullptr);
        EXPECT_TRUE(dataItem.CC2_p_ != nullptr);
        EXPECT_TRUE(dataItem.CC1_d_ != nullptr);
        EXPECT_TRUE(dataItem.CC2_d_ != nullptr);

        const amrex::Dim3 loGC  = dataItem.loGC();
        const amrex::Dim3 hiGC  = dataItem.hiGC();
        const amrex::Dim3 begin = loGC;
        const amrex::Dim3 end{hiGC.x+1, hiGC.y+1, hiGC.z+1};

        // TODO: We are assuming here that there was a conversion between
        // amrex::Real (in source) and double (device).
        amrex::Array4<double> f(dataItem.CC1_p_, begin, end, NUNKVAR);
        const amrex::Dim3 lbound = amrex::lbound(f);
        const amrex::Dim3 ubound = amrex::ubound(f);
        EXPECT_EQ(lbound.x, loGC.x);
        EXPECT_EQ(lbound.y, loGC.y);
        EXPECT_EQ(lbound.z, loGC.z);
        EXPECT_EQ(ubound.x, hiGC.x);
        EXPECT_EQ(ubound.y, hiGC.y);
        EXPECT_EQ(ubound.z, hiGC.z);

        for     (int j = loGC.y; j <= hiGC.y; ++j) {
            for (int i = loGC.x; i <= hiGC.x; ++i) {
                EXPECT_EQ(2.2, f(i, j, loGC.z, DENS_VAR_C));
                EXPECT_EQ(2.2, f(i, j, loGC.z, ENER_VAR_C));
//                EXPECT_EQ(i,       f(i, j, loGC.z, DENS_VAR_C));
//                EXPECT_EQ(2.0 * j, f(i, j, loGC.z, ENER_VAR_C));
            }
        }
    }

    // Transfer data to device and as it to double data
    // to confirm that device data pointers are correct
    cudaError_t   cErr = cudaErrorInvalidValue;
    std::cout << "Synchronous copy of " << packet.nBytes_
              << " bytes from pinned to device\n";
    cErr = cudaMemcpy(packet.start_d_, packet.start_p_,
                      packet.nBytes_,
                      cudaMemcpyHostToDevice);
    ASSERT_EQ(cudaSuccess, cErr);
//    std::cout << "Asynchronous copy of " << packet.nBytes_
//              << " bytes from pinned to device\n";
//    cErr = cudaMemcpyAsync(packet.devicePtr_, packet.pinnedPtr_,
//                           packet.nBytes_,
//                           cudaMemcpyHostToDevice,
//                           *(packet.stream_));
//    ASSERT_EQ(cudaSuccess, cErr);

    // Change data in device memory
    gpuKernel::kernel_packet(packet);

    // Bring data pack to pinned and confirm correct updated values
    std::cout << "Synchronous copy of " << packet.nBytes_
              << " bytes from device to pinned\n";
    cErr = cudaMemcpy(packet.start_p_, packet.start_d_,
                      packet.nBytes_,
                      cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaSuccess, cErr);
    for (std::size_t n=0; n<packet.nDataItems(); ++n) {
        const Tile&   dataItem = packet[n];

        EXPECT_TRUE(dataItem.CC_h_  != nullptr);
        ASSERT_TRUE(dataItem.CC1_p_ != nullptr);
        EXPECT_TRUE(dataItem.CC2_p_ != nullptr);
        EXPECT_TRUE(dataItem.CC1_d_ != nullptr);
        EXPECT_TRUE(dataItem.CC2_d_ != nullptr);

        const amrex::Dim3 loGC  = dataItem.loGC();
        const amrex::Dim3 hiGC  = dataItem.hiGC();
        const amrex::Dim3 begin = loGC;
        const amrex::Dim3 end{hiGC.x+1, hiGC.y+1, hiGC.z+1};

        // TODO: We are assuming here that there was a conversion between
        // amrex::Real (in source) and double (device).
        amrex::Array4<double> f(dataItem.CC1_p_, begin, end, NUNKVAR);
        const amrex::Dim3 lbound = amrex::lbound(f);
        const amrex::Dim3 ubound = amrex::ubound(f);
        EXPECT_EQ(lbound.x, loGC.x);
        EXPECT_EQ(lbound.y, loGC.y);
        EXPECT_EQ(lbound.z, loGC.z);
        EXPECT_EQ(ubound.x, hiGC.x);
        EXPECT_EQ(ubound.y, hiGC.y);
        EXPECT_EQ(ubound.z, hiGC.z);

        for     (int j = loGC.y; j <= hiGC.y; ++j) {
            for (int i = loGC.x; i <= hiGC.x; ++i) {
                EXPECT_EQ(2.2*copyInData.coefficient, f(i, j, loGC.z, DENS_VAR_C));
                EXPECT_EQ(2.2*copyInData.coefficient, f(i, j, loGC.z, ENER_VAR_C));
            }
        }
    }
}

}

