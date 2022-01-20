#include <gtest/gtest.h>

#include <Milhoja_Grid.h>
#include <Milhoja_Tile.h>

#include "Base.h"

using namespace milhoja;

namespace {

TEST(GridUnitTest,GCFill){
    Grid& grid = Grid::instance();
    float eps = 1.0e-14;

    grid.fillGuardCells();
    // Test Guard cell fill
    Real expected_val = 0.0_wp;
    for (int lev = 0; lev<=grid.getMaxLevel(); ++lev) {
    for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
        std::unique_ptr<Tile> tileDesc = ti->buildCurrentTile();

        RealVect cellCenter = tileDesc->getCenterCoords();

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

                    EXPECT_NEAR( expected_val, data(i,j,k,DENS_VAR), eps);
                }
            }
        }

    } //iterator loop
    } //level loop
}

}
