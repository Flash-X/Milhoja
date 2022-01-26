#include <gtest/gtest.h>

#include <Milhoja_Grid.h>
#include <Milhoja_Tile.h>

using namespace milhoja;

namespace {

TEST(GridUnitTest,MultipleLevels){
    Grid& grid = Grid::instance();
    float eps = 1.0e-10;

    for(auto ti=grid.buildTileIter(0); ti->isValid(); ti->next()) {
        auto tileDesc = ti->buildCurrentTile();
        auto data = tileDesc->data();
        auto lo = tileDesc->lo();
        auto hi = tileDesc->hi();
        for (int k = lo.K(); k <= hi.K(); ++k) {
        for (int j = lo.J(); j <= hi.J(); ++j) {
        for (int i = lo.I(); i <= hi.I(); ++i) {
            data(i,j,k,0) = 1.15_wp;
        }}}
    }

    grid.fillGuardCells();
    grid.updateGrid();

    for(auto ti=grid.buildTileIter(1); ti->isValid(); ti->next()) {
        auto tileDesc = ti->buildCurrentTile();
        auto data = tileDesc->data();
        auto lo = tileDesc->lo();
        auto hi = tileDesc->hi();
        for (int k = lo.K(); k <= hi.K(); ++k) {
        for (int j = lo.J(); j <= hi.J(); ++j) {
        for (int i = lo.I(); i <= hi.I(); ++i) {
            data(i,j,k,0) = 1.25_wp;
        }}}
    }
    grid.restrictAllLevels();

    grid.fillGuardCells();
    grid.updateGrid();

    for(auto ti=grid.buildTileIter(2); ti->isValid(); ti->next()) {
        auto tileDesc = ti->buildCurrentTile();
        auto data = tileDesc->data();
        auto lo = tileDesc->lo();
        auto hi = tileDesc->hi();
        for (int k = lo.K(); k <= hi.K(); ++k) {
        for (int j = lo.J(); j <= hi.J(); ++j) {
        for (int i = lo.I(); i <= hi.I(); ++i) {
            ASSERT_NEAR(data(i,j,k,0) , 1.25_wp, eps);
        }}}
    }
}

}

