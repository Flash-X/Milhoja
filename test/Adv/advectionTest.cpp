#include "Flash.h"
#include "constants.h"

#include "Grid.h"
#include "Grid_Macros.h"
#include "Grid_Edge.h"
#include "Grid_Axis.h"
#include "setInitialAdv.h"
#include "errorEstAdv.h"
#include "gtest/gtest.h"
#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include "Tile.h"
#include "Driver.h"


using namespace orchestration;

namespace {

//test fixture
class AdvectionTest : public testing::Test {
protected:
    AdvectionTest(void) {
    }

    ~AdvectionTest(void) {
            Grid::instance().destroyDomain();
    }
};

TEST_F(AdvectionTest,InitializeCheck){
    Grid& grid = Grid::instance();


    grid.initDomain(Simulation::setInitialAdv,
                    Simulation::errorEstAdv);

    //grid.writePlotfile("adv_plt_0000");
}

TEST_F(AdvectionTest,Evolution){
    Grid& grid = Grid::instance();

    grid.initDomain(Simulation::setInitialAdv,
                    Simulation::errorEstAdv);

    grid.writePlotfile("adv_plt_0000");

    Driver::EvolveAdvection();

    grid.writePlotfile("adv_plt_0001");
}

}
