#include "Grid.h"
#include "Flash.h"
#include "constants.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"
#include <AMReX.H>

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
        grid::Real realzero = 0.0_wp;
        grid::Real mypi = 3.14_wp;
        bool amrexReal_eq_gridReal = std::is_same<amrex::Real,grid::Real>::value;
        EXPECT_TRUE(amrexReal_eq_gridReal);
}

TEST_F(TestGrid,TestVectorClass){
        grid::Vector<grid::Real> realVec1 = {1.5_wp,3.2_wp,5.8_wp};
        grid::Vector<int> intVec1 = {3,10,2};

        grid::Vector<int> intVec2 = grid::Vector<int>(realVec1);
        EXPECT_TRUE(intVec2[0] == 1);
        EXPECT_TRUE(intVec2[1] == 3);
        EXPECT_TRUE(intVec2[2] == 5);
        grid::Vector<grid::Real> realVecSum = realVec1 + grid::Vector<grid::Real>(intVec1);
        EXPECT_TRUE(realVecSum[0] == 4.5_wp);
        EXPECT_TRUE(realVecSum[1] == 13.2_wp);
        EXPECT_TRUE(realVecSum[2] == 7.8_wp);
}

TEST_F(TestGrid,TestDomainBoundBox){
        Grid& grid = Grid::instance();
        grid::Vector<grid::Real> domainLo = grid.getDomainLo();
        grid::Vector<grid::Real> domainHi = grid.getDomainHi();

        EXPECT_TRUE(domainLo[0] == X_MIN );
        EXPECT_TRUE(domainLo[1] == Y_MIN );
        EXPECT_TRUE(domainLo[2] == Z_MIN );
        EXPECT_TRUE(domainHi[0] == X_MAX );
        EXPECT_TRUE(domainHi[1] == Y_MAX );
        EXPECT_TRUE(domainHi[2] == Z_MAX );
}

TEST_F(TestGrid,TestGetters){
        Grid& grid = Grid::instance();
        grid::Vector<grid::Real> domainLo = {X_MIN,Y_MIN,Z_MIN};
        grid::Vector<int> nBlocks = {N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z};
        grid::Vector<int> nCells = {NXB,NYB,NZB};

        //Testing Grid::getDeltas
        //TODO: loop over all levels when AMR is implemented
        grid::Vector<grid::Real> deltas = grid.getDeltas(0);
        grid::Real dx = (X_MAX - X_MIN) / static_cast<grid::Real>(N_BLOCKS_X * NXB);
        grid::Real dy = (Y_MAX - Y_MIN) / static_cast<grid::Real>(N_BLOCKS_Y * NYB);
        grid::Real dz = (Z_MAX - Z_MIN) / static_cast<grid::Real>(N_BLOCKS_Z * NZB);
        EXPECT_TRUE(dx == deltas[0]);
        EXPECT_TRUE(dy == deltas[1]);
        EXPECT_TRUE(dz == deltas[2]);

        //Testing Grid::getBlkCenterCoords
        for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
            Tile tileDesc(itor, 0);
            grid::Vector<grid::Real> sumVec = grid::Vector<grid::Real>( tileDesc.loVect()+tileDesc.hiVect());
            grid::Real x = X_MIN + dx * sumVec[0]/2.0;
            grid::Real y = Y_MIN + dy * sumVec[1]/2.0;
            grid::Real z = Z_MIN + dz * sumVec[2]/2.0;
            grid::Vector<grid::Real> blkCenterCoords = grid.getBlkCenterCoords(tileDesc);
            ASSERT_TRUE(x == blkCenterCoords[0]);
            ASSERT_TRUE(y == blkCenterCoords[1]);
            ASSERT_TRUE(z == blkCenterCoords[2]);
        }

        //Testing Grid::getMaxRefinement and getMaxLevel
        EXPECT_TRUE(0 == grid.getMaxRefinement());
        EXPECT_TRUE(0 == grid.getMaxLevel());
}

}
