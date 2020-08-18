#include "Flash.h"
#include "constants.h"

#include "Grid.h"
#include "Grid_Macros.h"
#include "Grid_Edge.h"
#include "Grid_Axis.h"
#include "setInitialConditions_block.h"
#include "setInitialInteriorTest.h"
#include "errorEstBlank.h"
#include "errorEstMaximal.h"
#include "gtest/gtest.h"
#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include "Tile.h"



using namespace orchestration;

namespace {

//test fixture
class GridAMRTest : public testing::Test {
protected:
    GridAMRTest(void) {
    }

    ~GridAMRTest(void) {
            Grid::instance().destroyDomain();
    }
};

TEST_F(GridAMRTest,Initialization){
    Grid& grid = Grid::instance();
    float eps = 1.0e-14;

    grid.initDomain(Simulation::setInitialConditions_block,
                    Simulation::errorEstMaximal);

    IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};

    // Testing Grid::getMaxRefinement and getMaxLevel
    EXPECT_EQ(grid.getMaxRefinement() , LREFINE_MAX-1);
    EXPECT_EQ(grid.getMaxLevel()      , LREFINE_MAX-1);
}

TEST_F(GridAMRTest,GCFill){
    Grid& grid = Grid::instance();
    float eps = 1.0e-14;

    grid.initDomain(Simulation::setInitialInteriorTest,
                    Simulation::errorEstBlank);
    EXPECT_EQ(grid.getMaxLevel()      , 0);

    // Test Guard cell fill
    Real expected_val = 0.0_wp;
    for (int lev = 0; lev<=grid.getMaxLevel(); ++lev) {
    for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
        std::unique_ptr<Tile> tileDesc = ti->buildCurrentTile();

        RealVect cellCenter = tileDesc->getCenterCoords();
        std::cout << "Testing tile with center coords: " << cellCenter;
        std::cout << std::endl;

        IntVect lo = tileDesc->lo();
        IntVect hi = tileDesc->hi();
        IntVect loGC = tileDesc->loGC();
        IntVect hiGC = tileDesc->hiGC();
        FArray4D data = tileDesc->data();
        for (        int k = loGC.K(); k <= hiGC.K(); ++k) {
            for (    int j = loGC.J(); j <= hiGC.J(); ++j) {
                for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                    IntVect pos{LIST_NDIM(i,j,k)};
                    if (pos.allGE(lo) && pos.allLE(hi) ) {
                        continue;
                    }

                    expected_val  = 1.0_wp;

                    std::cout << "Testing GC at ";
                    std::cout << IntVect(LIST_NDIM(i,j,k)) << std::endl;

                    EXPECT_NEAR( expected_val, data(i,j,k,DENS_VAR_C), eps);
                }
            }
        }

    } //iterator loop
    } //level loop
}

}
