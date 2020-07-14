#include "Grid.h"
#include "Grid_Macros.h"
#include "Flash.h"
#include "constants.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"

#include <iostream>

using namespace orchestration;

namespace {

//test fixture
class GridTimingTest : public testing::Test {
protected:
        GridTimingTest(void) {
                RealVect probMin{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
                RealVect probMax{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
                IntVect  nBlocks{LIST_NDIM(N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z)};
                Grid&    grid = Grid::instance();
                grid.initDomain(probMin,probMax,nBlocks,NUNKVAR,
                                Simulation::setInitialConditions_block);
        }

        ~GridTimingTest(void) {
                Grid::instance().destroyDomain();
        }
};

TEST_F(GridTimingTest,VectorDirectInitialization){
        using namespace orchestration;
        //test creation and conversion
        IntVect expected = IntVect(LIST_NDIM(3,10,2));
        for (int i=0;i<=1000000;i++){
          IntVect intVec1{LIST_NDIM(3,10,2)};
          EXPECT_TRUE( intVec1 == expected );
        }
}

TEST_F(GridTimingTest,VectorCopyInitialization){
        using namespace orchestration;
        //test creation and conversion
        IntVect expected = IntVect(LIST_NDIM(3,10,2));
        for (int i=0;i<=1000000;i++){
          IntVect intVec1 = IntVect(LIST_NDIM(3,10,2));
          EXPECT_TRUE( intVec1 == expected );
        }
}

TEST_F(GridTimingTest,TestGetVolumeSingle){
        float eps = 1.0e-14;

        Grid& grid = Grid::instance();
        RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
        RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
        IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
        IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
        RealVect deltas_actual = (actual_max-actual_min) / RealVect(nBlocks*nCells);
        Real actual_vol = deltas_actual.product();

        // Create and initialize C++ array to store volumes
        amrex::Geometry&  geom = grid.geometry();
        amrex::IntVect vlo = geom.Domain().smallEnd();
        amrex::IntVect vhi = geom.Domain().bigEnd();

        if ( (vhi[0]-vlo[0]+1) * (vhi[1]-vlo[1]+1) > 1023*1023)
            throw std::logic_error("domain too large, can't declare c++ array to store volumes");

        Real volumes[ vhi[0]-vlo[0]+1 ][ vhi[1]-vlo[1]+1 ];
        for (int i=0; i<=(vhi-vlo)[0]; ++i){
            for (int j=0; j<=(vhi-vlo)[1]; ++j) {
                volumes[vlo[0]+i][vlo[1]+j] = 0.0;
            }
        }

        // Fill volume array cell-by-cell
        for (amrex::MFIter itor(grid.unk(),amrex::IntVect(1)); itor.isValid(); ++itor) {
            Tile tileDesc(itor, 0);
            IntVect coord =  tileDesc.loVect();
            volumes[ coord[0]-vlo[0] ][ coord[1]-vlo[1] ] = grid.getCellVolume(0,coord);
        }

        // Check all volumes are correct
        for (int i=0; i<=(vhi-vlo)[0]; ++i){
            for (int j=0; j<=(vhi-vlo)[1]; ++j) {
                ASSERT_NEAR( actual_vol , volumes[vlo[0]+i][vlo[1]+j] , eps);
            }
        }
}

TEST_F(GridTimingTest,TestGetVolumeSingle2){
        float eps = 1.0e-14;

        Grid& grid = Grid::instance();
        RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
        RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
        IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
        IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
        RealVect deltas_actual = (actual_max-actual_min) / RealVect(nBlocks*nCells);
        Real actual_vol = deltas_actual.product();

        for (amrex::MFIter itor(grid.unk(),amrex::IntVect(1)); itor.isValid(); ++itor) {
            Tile tileDesc(itor, 0);
            IntVect coord =  tileDesc.loVect();
            ASSERT_NEAR( actual_vol , grid.getCellVolume(0,coord) , eps);
        }
}

TEST_F(GridTimingTest,TestGetVolumeMulti){
        float eps = 1.0e-14;

        Grid& grid = Grid::instance();

        RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
        RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
        IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
        IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
        RealVect deltas_actual = (actual_max-actual_min) / RealVect(nBlocks*nCells);
        Real vol_actual = deltas_actual.product();

        // Create a FArrayBox which will store the volumes
        amrex::Geometry&  geom = grid.geometry();
        amrex::IntVect vlo = geom.Domain().smallEnd();
        amrex::IntVect vhi = geom.Domain().bigEnd();
        amrex::Box bx(vlo,vhi);
        amrex::FArrayBox vol_fab(bx,1);
        Real* vol_ptr = vol_fab.dataPtr();

        // Use fillCellVolumes to fill the fab with all volumes
        grid.fillCellVolumes(0,IntVect(vlo[0],vlo[1]),IntVect(vhi[0],vhi[1]),vol_ptr);

        // Check all volumes are correct
        for (int i=0; i<=(vhi-vlo)[0]; ++i){
            for (int j=0; j<=(vhi-vlo)[1]; ++j) {
                EXPECT_NEAR( vol_fab({vlo[0]+i,vlo[1]+j},0) , vol_actual, eps);
            }
        }

}


}
