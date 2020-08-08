#include "Flash.h"
#include "constants.h"

#include "Grid.h"
#include "Grid_Macros.h"
#include "Grid_Edge.h"
#include "Grid_Axis.h"
#include "setInitialAdv.h"
#include "gtest/gtest.h"
#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include "Tile.h"

// Macro for iterating over all coordinates in the
// region defined by two IntVects lo and hi.
// Middle three arguments are the iteration variables,
// which can be used in 'function'.

#define ITERATE_REGION(lo,hi,i,j,k, function) {\
for(int i=lo.I();i<=hi.I();++i) {\
for(int j=lo.J();j<=hi.J();++j) {\
for(int k=lo.K();k<=hi.K();++k) {\
    function \
}}}}


using namespace orchestration;

namespace {

//test fixture
class GridUnitTest : public testing::Test {
protected:
    GridUnitTest(void) {
    }

    ~GridUnitTest(void) {
            Grid::instance().destroyDomain();
    }
};

TEST_F(GridUnitTest,InitializeCheck){
    using namespace orchestration;
    Grid& grid = Grid::instance();


    grid.initDomain(Simulation::setInitialAdv);

    grid.writePlotfile("adv_plt_0000");
}

}
