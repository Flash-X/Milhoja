#include "Flash.h"
#include "constants.h"

#include "Grid.h"
#include "Grid_Macros.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"
#include <AMReX.H>

using namespace orchestration;

namespace {

//test fixture
class GridUnitTest : public testing::Test {
protected:
        GridUnitTest(void) {
                RealVect probMin{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
                RealVect probMax{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
                IntVect  nBlocks{LIST_NDIM(N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z)};
                Grid&    grid = Grid::instance();
                grid.initDomain(probMin,probMax,nBlocks,NUNKVAR,
                                Simulation::setInitialConditions_block);
        }

        ~GridUnitTest(void) {
                Grid::instance().destroyDomain();
        }
};

TEST_F(GridUnitTest,TestVectorClasses){
        using namespace orchestration;
        //test creation and conversion
        IntVect intVec1{LIST_NDIM(3,10,2)};
        RealVect realVec1{LIST_NDIM(1.5_wp,3.2_wp,5.8_wp)};
        IntVect intVec2 = IntVect(realVec1);
        RealVect realVec2 = RealVect(intVec1);

        //test operators for IntVect
        EXPECT_TRUE( intVec2 == IntVect(LIST_NDIM(1,3,5)) );
        EXPECT_TRUE( intVec1 != intVec2 );
        EXPECT_TRUE( intVec1+intVec2 == IntVect(LIST_NDIM(4,13,7)) );
        EXPECT_TRUE( intVec1-intVec2 == IntVect(LIST_NDIM(2,7,-3)) );
        EXPECT_TRUE( intVec1*intVec2 == IntVect(LIST_NDIM(3,30,10)) );
        EXPECT_TRUE( intVec1+5 == IntVect(LIST_NDIM(8,15,7)) );
        EXPECT_TRUE( intVec1-9 == IntVect(LIST_NDIM(-6,1,-7)) );
        EXPECT_TRUE( intVec1*2 == IntVect(LIST_NDIM(6,20,4)) );
        EXPECT_TRUE( 2*intVec1 == IntVect(LIST_NDIM(6,20,4)) );
        EXPECT_TRUE( intVec1/2 == IntVect(LIST_NDIM(1,5,1)) );
        EXPECT_TRUE( intVec1.product() == CONCAT_NDIM(3,*10,*2) );

        //test operators for RealVect
        float eps = 1.0e-14;
        for (int i=0;i<NDIM;++i) {
            EXPECT_NEAR( realVec2[i] , RealVect(LIST_NDIM(3.0_wp,10.0_wp,2.0_wp))[i] , eps );
            EXPECT_NEAR( (realVec1+realVec2)[i] , RealVect(LIST_NDIM(4.5_wp,13.2_wp,7.8_wp))[i] , eps);
            EXPECT_NEAR( (realVec1-realVec2)[i] , RealVect(LIST_NDIM(-1.5_wp,-6.8_wp,3.8_wp))[i] , eps);
            EXPECT_NEAR( (realVec1*realVec2)[i] , RealVect(LIST_NDIM(4.5_wp,32.0_wp,11.6_wp))[i] , eps);
            EXPECT_NEAR( (realVec1*-3.14_wp)[i] , RealVect(LIST_NDIM(-4.71_wp,-10.048_wp,-18.212_wp))[i] , eps);
            EXPECT_NEAR( (-3.14_wp*realVec1)[i] , RealVect(LIST_NDIM(-4.71_wp,-10.048_wp,-18.212_wp))[i] , eps);
            EXPECT_NEAR( (realVec1/2.0_wp)[i] , RealVect(LIST_NDIM(0.75_wp,1.6_wp,2.9_wp))[i] , eps);
        }
}

TEST_F(GridUnitTest,TestGetters){
        float eps = 1.0e-14;
        int count;

        Grid& grid = Grid::instance();
        RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
        RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
        IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
        IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
        RealVect deltas_actual = (actual_max-actual_min) / RealVect(nBlocks*nCells);

        // Testing Grid::getDomain{Lo,Hi}
        RealVect domainLo = grid.getDomainLo();
        RealVect domainHi = grid.getDomainHi();
        for (int i=0;i<NDIM;++i) {
            EXPECT_NEAR(domainLo[i] , actual_min[i] , eps);
            EXPECT_NEAR(domainHi[i] , actual_max[i] , eps);
        }

        // Testing Grid::getDeltas
        //TODO: loop over all levels when AMR is implemented
        RealVect deltas = grid.getDeltas(0);
        for(int i=1;i<NDIM;++i) {
            EXPECT_NEAR(deltas_actual[i] , deltas[i], eps);
        }

        // Test Grid::getBlkCenterCoords with block iterator
        count = 0;
        for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
            count++;
            if(count%3 != 0) continue;

            Tile tileDesc(itor, 0);
            RealVect sumVec = RealVect(tileDesc.lo()+tileDesc.hi()+1);
            RealVect coords = actual_min + deltas_actual*sumVec*0.5_wp;

            RealVect blkCenterCoords = grid.getBlkCenterCoords(tileDesc);
            for(int i=1;i<NDIM;++i) {
                ASSERT_NEAR(coords[i] , blkCenterCoords[i], eps);
            }
        }

        // Test Grid::getCellVolume and Grid::getCellFaceArea with cell-by-cell iterator
        Real actual_vol = deltas_actual.product();
        RealVect actual_fa;
        for (int i=0;i<NDIM;++i) {
            int p1 = int(i==0);
            int p2 = 2 - int(i==2);
            actual_fa[i] = CONCAT_NDIM( 1.0_wp, *deltas_actual[p1], *deltas_actual[p2] );
        }
        count = 0;
        for (amrex::MFIter itor(grid.unk(),amrex::IntVect(1)); itor.isValid(); ++itor) {
            count++;
            if(count%7 != 0) continue;
            Tile tileDesc(itor, 0);
            IntVect coord = tileDesc.lo();
            ASSERT_NEAR( actual_vol , grid.getCellVolume(0,coord) , eps);
            for(int i=1;i<NDIM;++i) {
                ASSERT_NEAR( actual_fa[i] , grid.getCellFaceArea(0,i,coord) , eps);
            }
        }


        // Test Grid::fillCellVolumes over an arbitrary range
        IntVect vlo{LIST_NDIM(1,2,3)};
        IntVect vhi{LIST_NDIM(2,4,7)};
        amrex::Box bx{ amrex::IntVect(vlo), amrex::IntVect(vhi) };
        amrex::FArrayBox vol_fab{bx,1};
        Real* vol_ptr = vol_fab.dataPtr();
        grid.fillCellVolumes(0,vlo,vhi,vol_ptr);
#if NDIM==2
        for (int i=vlo[0]; i<=vhi[0]; ++i){
            for (int j=vlo[1]; j<=vhi[1]; ++j) {
                EXPECT_NEAR( vol_fab({i,j},0) , actual_vol , eps);
            }
        }
#endif
#if NDIM==3
        for (int i=vlo[0]; i<=vhi[0]; ++i){
            for (int j=vlo[1]; j<=vhi[1]; ++j) {
                for (int k=vlo[2]; k<=vhi[2]; ++k) {
                    EXPECT_NEAR( vol_fab({i,j,k},0) , actual_vol , eps);
                }
            }
        }
#endif

        //Testing Grid::getMaxRefinement and getMaxLevel
        EXPECT_TRUE(0 == grid.getMaxRefinement());
        EXPECT_TRUE(0 == grid.getMaxLevel());
}

}
