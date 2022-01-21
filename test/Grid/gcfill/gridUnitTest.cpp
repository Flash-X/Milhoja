#include <limits>

#include <gtest/gtest.h>

#include <Milhoja_Grid.h>
#include <Milhoja_Tile.h>

#include "Base.h"

namespace {

TEST(GridUnitTest, GCFill) {
    using namespace milhoja;

    // Since the expected data and the data used to perform the GC fill for the
    // density data can be exactly represented in double precision, we can
    // insist on equality.
    constexpr double   EPS = std::numeric_limits<double>::epsilon();
    constexpr double   THRESHOLD = 0.0 * EPS;

    Grid& grid = Grid::instance();

    // TODO: Confirm that the GC are not set prior to calling the fill.

    grid.fillGuardCells();

    // Test Guard cell fill
    for (unsigned int level=0; level<=grid.getMaxLevel(); ++level) {
        for (auto ti = grid.buildTileIter(level); ti->isValid(); ti->next()) {
            std::unique_ptr<Tile> tileDesc = ti->buildCurrentTile();

            IntVect loGC  = tileDesc->loGC();
            IntVect hiGC  = tileDesc->hiGC();
            FArray4D data = tileDesc->data();
            for (        int k = loGC.K(); k <= hiGC.K(); ++k) {
                for (    int j = loGC.J(); j <= hiGC.J(); ++j) {
                    for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                        // TODO: Add in check on ENER_VAR since this data varies
                        // across blocks.
                        EXPECT_NEAR(1.0_wp, data(i, j, k, DENS_VAR), THRESHOLD);
                    }
                }
            }

        }
    }
}

}

