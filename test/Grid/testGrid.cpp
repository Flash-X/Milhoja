#include "Grid.h"
#include "Flash.h"
#include "constants.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"
#include <AMReX.H>

using namespace orchestration;

namespace {

//test fixture
class TestGrid : public testing::Test {
protected:
        TestGrid(void) {
                Grid&    grid = Grid::instance();
                grid.initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                  N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                  NUNKVAR,Simulation::setInitialConditions_block);
        }

        ~TestGrid(void) {
                Grid::instance().destroyDomain();
        }
};

TEST_F(TestGrid,TestVectorClasses){
        using namespace orchestration;
        //test creation and conversion
        IntVect intVec1 = IntVect(3,10,2);
        RealVect realVec1 = RealVect(1.5_wp,3.2_wp,5.8_wp);
        IntVect intVec2 = IntVect(realVec1);
        RealVect realVec2 = RealVect(intVec1);

        EXPECT_TRUE( intVec2 == IntVect(1,3,5) );
        EXPECT_TRUE( realVec2 == RealVect(3_wp,10_wp,2_wp) );

        //test operators for IntVect
        EXPECT_TRUE( intVec1 != intVec2 );
        EXPECT_TRUE( intVec1+intVec2 == IntVect(4,13,7) );
        EXPECT_TRUE( intVec1-intVec2 == IntVect(2,7,-3) );
        EXPECT_TRUE( intVec1*intVec2 == IntVect(3,30,10) );
        EXPECT_TRUE( intVec1*2 == IntVect(6,20,4) );
        EXPECT_TRUE( 2*intVec1 == IntVect(6,20,4) );
        EXPECT_TRUE( intVec1/2 == IntVect(1,5,1) );

        //test operators for RealVect
        EXPECT_TRUE( realVec1 != realVec2 );
        EXPECT_TRUE( realVec1+realVec2 == RealVect(4.5_wp,13.2_wp,7.8_wp) );
        EXPECT_TRUE( realVec1-realVec2 == RealVect(-1.5_wp,-6.8_wp,3.8_wp) );
        EXPECT_TRUE( realVec1*realVec2 == RealVect(4.5_wp,32_wp,11.6_wp) );
        EXPECT_TRUE( realVec1*-3.14_wp == RealVect(-4.71_wp,-10.048_wp,-18.212_wp) );
        EXPECT_TRUE( -3.14_wp*realVec1 == RealVect(-4.71_wp,-10.048_wp,-18.212_wp) );
        EXPECT_TRUE( realVec1/2_wp == RealVect(.75_wp,1.6_wp,2.9_wp) );
}

TEST_F(TestGrid,TestDomainBoundBox){
        Grid& grid = Grid::instance();
        Vector<Real> domainLo = grid.getDomainLo();
        Vector<Real> domainHi = grid.getDomainHi();

        EXPECT_TRUE(domainLo[0] == X_MIN );
        EXPECT_TRUE(domainLo[1] == Y_MIN );
        EXPECT_TRUE(domainLo[2] == Z_MIN );
        EXPECT_TRUE(domainHi[0] == X_MAX );
        EXPECT_TRUE(domainHi[1] == Y_MAX );
        EXPECT_TRUE(domainHi[2] == Z_MAX );
}

TEST_F(TestGrid,TestGetters){
        Grid& grid = Grid::instance();
        Vector<Real> domainLo = {X_MIN,Y_MIN,Z_MIN};
        Vector<int> nBlocks = {N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z};
        Vector<int> nCells = {NXB,NYB,NZB};

        //Testing Grid::getDeltas
        //TODO: loop over all levels when AMR is implemented
        Vector<Real> deltas = grid.getDeltas(0);
        Real dx = (X_MAX - X_MIN) / static_cast<Real>(N_BLOCKS_X * NXB);
        Real dy = (Y_MAX - Y_MIN) / static_cast<Real>(N_BLOCKS_Y * NYB);
        Real dz = (Z_MAX - Z_MIN) / static_cast<Real>(N_BLOCKS_Z * NZB);
        EXPECT_TRUE(dx == deltas[0]);
        EXPECT_TRUE(dy == deltas[1]);
        EXPECT_TRUE(dz == deltas[2]);

        //Testing Grid::getBlkCenterCoords
        for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
            Tile tileDesc(itor, 0);
            Vector<Real> sumVec = Vector<Real>( tileDesc.loVect()+tileDesc.hiVect());
            Real x = X_MIN + dx * sumVec[0]*0.5_wp;
            Real y = Y_MIN + dy * sumVec[1]*0.5_wp;
            Real z = Z_MIN + dz * sumVec[2]*0.5_wp;
            Vector<Real> blkCenterCoords = grid.getBlkCenterCoords(tileDesc);
            ASSERT_TRUE(x == blkCenterCoords[0]);
            ASSERT_TRUE(y == blkCenterCoords[1]);
            ASSERT_TRUE(z == blkCenterCoords[2]);
        }

        //Testing Grid::getMaxRefinement and getMaxLevel
        EXPECT_TRUE(0 == grid.getMaxRefinement());
        EXPECT_TRUE(0 == grid.getMaxLevel());
}

}
