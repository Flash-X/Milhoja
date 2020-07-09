#include "Grid.h"
#include "Flash.h"
#include "constants.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"

using namespace orchestration;

namespace {

//test fixture
class GridTimingTest : public testing::Test {
protected:
        GridTimingTest(void) {
                //Grid&    grid = Grid::instance();
                //grid.initDomain(X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
                //  N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z,
                 // NUNKVAR,Simulation::setInitialConditions_block);
        }

        ~GridTimingTest(void) {
                Grid::instance().destroyDomain();
        }
};

TEST_F(GridTimingTest,VectorDirectInitialization){
        using namespace orchestration;
        //test creation and conversion
        IntVect expected = IntVect(3,10,2);
        for (int i=0;i<=1000000;i++){
          IntVect intVec1{3,10,2};
          EXPECT_TRUE( intVec1 == expected );
        }
}

TEST_F(GridTimingTest,VectorCopyInitialization){
        using namespace orchestration;
        //test creation and conversion
        IntVect expected = IntVect(3,10,2);
        for (int i=0;i<=1000000;i++){
          IntVect intVec1 = IntVect(3,10,2);
          EXPECT_TRUE( intVec1 == expected );
        }
}


}
