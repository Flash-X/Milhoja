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

TEST_F(TestGrid,TestSample){
        EXPECT_TRUE(0==0);
}

TEST_F(TestGrid,TestRealTypeDef){
        Real realzero = 0.0_wp;
        Real mypi = 3.14_wp;
        bool amrexReal_eq_gridReal = std::is_same<amrex::Real,Real>::value;
        EXPECT_TRUE(amrexReal_eq_gridReal);
}

TEST_F(TestGrid,TestVectorClass){
        Vector<Real> realVec1 = {1.5_wp,3.2_wp,5.8_wp};
        Vector<int> intVec1 = {3,10,2};

        Vector<int> intVec2 = Vector<int>(realVec1);
        EXPECT_TRUE(intVec2[0] == 1);
        EXPECT_TRUE(intVec2[1] == 3);
        EXPECT_TRUE(intVec2[2] == 5);
        Vector<Real> realVecSum = realVec1 + Vector<Real>(intVec1);
        EXPECT_TRUE(realVecSum[0] == 4.5_wp);
        EXPECT_TRUE(realVecSum[1] == 13.2_wp);
        EXPECT_TRUE(realVecSum[2] == 7.8_wp);
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
            Real x = X_MIN + dx * sumVec[0]/2.0;
            Real y = Y_MIN + dy * sumVec[1]/2.0;
            Real z = Z_MIN + dz * sumVec[2]/2.0;
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
