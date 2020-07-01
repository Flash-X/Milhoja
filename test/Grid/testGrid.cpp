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
        grid::Real realzero = 0_rt;
        grid::Real mypi = 3.14_rt;
        bool amrexReal_eq_gridReal = std::is_same<amrex::Real,grid::Real>::value;
        EXPECT_TRUE(amrexReal_eq_gridReal);
}

TEST_F(TestGrid,TestDomainBoundBox){
        Grid& grid = Grid::instance();
        std::vector<grid::Real> domainLo = grid.getDomainLo();
        std::vector<grid::Real> domainHi = grid.getDomainHi();

        EXPECT_TRUE(domainLo[0] == X_MIN );
        EXPECT_TRUE(domainLo[1] == Y_MIN );
        EXPECT_TRUE(domainLo[2] == Z_MIN );
        EXPECT_TRUE(domainHi[0] == X_MAX );
        EXPECT_TRUE(domainHi[1] == Y_MAX );
        EXPECT_TRUE(domainHi[2] == Z_MAX );
}

TEST_F(TestGrid,TestGetters){
        Grid& grid = Grid::instance();

        //Testing Grid::getDeltas
        //TODO: loop over all levels when AMR is implemented
        std::vector<grid::Real> deltas = grid.getDeltas(0);
        grid::Real dx = (X_MAX - X_MIN) / static_cast<grid::Real>(N_BLOCKS_X * NXB);
        grid::Real dy = (Y_MAX - Y_MIN) / static_cast<grid::Real>(N_BLOCKS_Y * NYB);
        grid::Real dz = (Z_MAX - Z_MIN) / static_cast<grid::Real>(N_BLOCKS_Z * NZB);
        EXPECT_TRUE(dx == deltas[0]);
        EXPECT_TRUE(dy == deltas[1]);
        EXPECT_TRUE(dz == deltas[2]);

        //Testing Grid::getBlkCenterCoords
        for (amrex::MFIter  itor(grid.unk()); itor.isValid(); ++itor) {
            Tile tileDesc(itor, 0);
            std::vector<int> loV = tileDesc.loVect();
            std::vector<int> hiV = tileDesc.hiVect();
            grid::Real x = X_MIN + dx * static_cast<grid::Real>(loV[0]+hiV[0]) / 2.0;
            grid::Real y = Y_MIN + dy * static_cast<grid::Real>(loV[1]+hiV[1]) / 2.0;
            grid::Real z = Z_MIN + dz * static_cast<grid::Real>(loV[2]+hiV[2]) / 2.0;
            std::vector<grid::Real> blkCenterCoords = grid.getBlkCenterCoords(tileDesc);
            ASSERT_TRUE(x == blkCenterCoords[0]);
            ASSERT_TRUE(y == blkCenterCoords[1]);
            ASSERT_TRUE(z == blkCenterCoords[2]);
        }

        //Testing Grid::getMaxRefinement and getMaxLevel
        EXPECT_TRUE(0 == grid.getMaxRefinement());
        EXPECT_TRUE(0 == grid.getMaxLevel());
}

}
