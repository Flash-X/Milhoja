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
                Grid&    grid = Grid::instance();
                grid.initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                  N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                  NUNKVAR,Simulation::setInitialConditions_block);
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
        std::cout << "Test print of intVec1: " << intVec1 << std::endl;
        EXPECT_TRUE( intVec2 == IntVect(LIST_NDIM(1,3,5)) );
        EXPECT_TRUE( intVec1 != intVec2 );
        EXPECT_TRUE( intVec1+intVec2 == IntVect(LIST_NDIM(4,13,7)) );
        EXPECT_TRUE( intVec1-intVec2 == IntVect(LIST_NDIM(2,7,-3)) );
        EXPECT_TRUE( intVec1*intVec2 == IntVect(LIST_NDIM(3,30,10)) );
        EXPECT_TRUE( intVec1*2 == IntVect(LIST_NDIM(6,20,4)) );
        EXPECT_TRUE( 2*intVec1 == IntVect(LIST_NDIM(6,20,4)) );
        EXPECT_TRUE( intVec1/2 == IntVect(LIST_NDIM(1,5,1)) );

        //test operators for RealVect
        float eps = 1.0e-14;
        std::cout << "Test print of realVec1: " << realVec1 << std::endl;
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

TEST_F(GridUnitTest,TestDomainBoundBox){
        Grid& grid = Grid::instance();
        RealVect domainLo = grid.getDomainLo();
        RealVect domainHi = grid.getDomainHi();
        RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
        RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};

        float eps = 1.0e-14;
        for (int i=0;i<NDIM;++i) {
            EXPECT_NEAR(domainLo[i] , actual_min[i] , eps);
            EXPECT_NEAR(domainHi[i] , actual_max[i] , eps);
        }
}

TEST_F(GridUnitTest,TestGetters){
        float eps = 1.0e-14;

        Grid& grid = Grid::instance();
        RealVect domainLo{LIST_NDIM(X_MIN, Y_MIN, Z_MIN)};
        IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
        IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};

        //Testing Grid::getDeltas
        //TODO: loop over all levels when AMR is implemented
        RealVect deltas = grid.getDeltas(0);
        Real actual_x,actual_y,actual_z;
        if(NDIM>=1) actual_x = (X_MAX - X_MIN) / Real(N_BLOCKS_X * NXB);
        if(NDIM>=2) actual_y = (Y_MAX - Y_MIN) / Real(N_BLOCKS_Y * NYB);
        if(NDIM>=3) actual_z = (Z_MAX - Z_MIN) / Real(N_BLOCKS_Z * NZB);
        RealVect deltas_actual{LIST_NDIM(actual_x, actual_y, actual_z)};
        for(int i=1;i<NDIM;++i) {
            EXPECT_NEAR(deltas_actual[i] , deltas[i], eps);
        }

        //Testing Grid::getBlkCenterCoords
        for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
            Tile tileDesc(itor, 0);
            RealVect sumVec = RealVect(tileDesc.loVect()+tileDesc.hiVect());
            RealVect coords = domainLo + deltas_actual*sumVec*0.5_wp;

            RealVect blkCenterCoords = grid.getBlkCenterCoords(tileDesc);

            for(int i=1;i<NDIM;++i) {
                ASSERT_NEAR(coords[i] , blkCenterCoords[i], eps);
            }
        }

        //Testing Grid::getMaxRefinement and getMaxLevel
        EXPECT_TRUE(0 == grid.getMaxRefinement());
        EXPECT_TRUE(0 == grid.getMaxLevel());
}

}
