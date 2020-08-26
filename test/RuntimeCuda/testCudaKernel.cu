#include "Tile.h"
#include "Grid.h"
#include "FArray4D.h"

#include "Flash.h"
#include "constants.h"

#include "gtest/gtest.h"

namespace {

void kernel_block(orchestration::FArray4D& f, 
                  const orchestration::IntVect& loGC,
                  const orchestration::IntVect& hiGC, 
                  const int i, const int j, const int k) {
    f(i, j, k, DENS_VAR_C) +=  2.1 * j;
    f(i, j, k, ENER_VAR_C) -=        i;
}

/**
 * Define a test fixture
 */
class TestCudaKernel : public testing::Test {
protected:
    static constexpr unsigned int   LEVEL = 0;

    static void setInitialConditions_block(const int tId, void* dataItem) {
        using namespace orchestration;

        Tile*  tileDesc = static_cast<Tile*>(dataItem);

        Grid&    grid = Grid::instance();

        const IntVect   loGC = tileDesc->loGC();
        const IntVect   hiGC = tileDesc->hiGC();
        FArray4D        f    = tileDesc->data();

        for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
            for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
                for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                    f(i, j, k, DENS_VAR_C) = i;
                    f(i, j, k, ENER_VAR_C) = 2.0 * j;
                }
            }
        }
    }

    TestCudaKernel(void) {
        orchestration::Grid&    grid = orchestration::Grid::instance();
        grid.initDomain(TestCudaKernel::setInitialConditions_block);
   }

    ~TestCudaKernel(void) {
        orchestration::Grid::instance().destroyDomain();
    }
};

TEST_F(TestCudaKernel, TestKernel) {
    using namespace orchestration;

    Grid&    grid = Grid::instance();

    // Run the kernel in the CPU at first
    for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next()) {
        std::unique_ptr<Tile>   tileDesc = ti->buildCurrentTile();

        const IntVect   loGC = tileDesc->loGC();
        const IntVect   hiGC = tileDesc->hiGC();
        FArray4D        f    = tileDesc->data();

        for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
            for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
                for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                    kernel_block(f, loGC, hiGC, i, j, k);
                }
            }
        }
    }

    // Check that kernel ran correctly    
    for (auto ti = grid.buildTileIter(LEVEL); ti->isValid(); ti->next()) {
        std::unique_ptr<Tile>   tileDesc = ti->buildCurrentTile();

        const IntVect   loGC = tileDesc->loGC();
        const IntVect   hiGC = tileDesc->hiGC();
        const FArray4D  f    = tileDesc->data();

        for         (int k = loGC.K(); k <= hiGC.K(); ++k) {
            for     (int j = loGC.J(); j <= hiGC.J(); ++j) {
                for (int i = loGC.I(); i <= hiGC.I(); ++i) {
                    EXPECT_EQ( i + 2.1 * j, f(i, j, k, DENS_VAR_C));
                    EXPECT_EQ(-i + 2.0 * j, f(i, j, k, ENER_VAR_C));
                }
            }
        }
    }
}

}

