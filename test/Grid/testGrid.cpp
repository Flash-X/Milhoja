#include "Grid.h"
#include "Flash.h"
#include "constants.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"
#include <AMReX_RealBox.H>

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

TEST_F(TestGrid,TestDomainBoundBox){
        double bbox[2][3];
        Grid::instance().getDomainBoundBox(bbox);

        EXPECT_TRUE(bbox[0][0] == X_MIN );
        EXPECT_TRUE(bbox[0][1] == Y_MIN );
        EXPECT_TRUE(bbox[0][2] == Z_MIN );
        EXPECT_TRUE(bbox[1][0] == X_MAX );
        EXPECT_TRUE(bbox[1][1] == Y_MAX );
        EXPECT_TRUE(bbox[1][2] == Z_MAX );
}

}
