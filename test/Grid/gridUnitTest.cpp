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
class GridUnitTest : public testing::Test {
protected:
    GridUnitTest(void) {
    }

    ~GridUnitTest(void) {
            Grid::instance().destroyDomain();
    }
};

TEST_F(GridUnitTest,VectorClasses){
    Grid::instance().initDomain(Simulation::setInitialConditions_block,
                                Simulation::errorEstBlank);

    //test creation and conversion
    IntVect intVec1{LIST_NDIM(3,10,2)};
    RealVect realVec1{LIST_NDIM(1.5_wp,3.2_wp,5.8_wp)};
    IntVect intVec2 = realVec1.floor();
    RealVect realVec2 = RealVect(intVec1);

    //test operators for IntVect
    EXPECT_EQ( realVec1.round()  , IntVect(LIST_NDIM(2,3,6)) );
    EXPECT_EQ( realVec1.floor()  , IntVect(LIST_NDIM(1,3,5)) );
    EXPECT_EQ( realVec1.ceil()   , IntVect(LIST_NDIM(2,4,6)) );
    EXPECT_NE( intVec1           , intVec2 );
    EXPECT_EQ( intVec1+intVec2   , IntVect(LIST_NDIM(4,13,7)) );
    EXPECT_EQ( intVec1-intVec2   , IntVect(LIST_NDIM(2,7,-3)) );
    EXPECT_EQ( intVec1*intVec2   , IntVect(LIST_NDIM(3,30,10)) );
    EXPECT_EQ( intVec1+5         , IntVect(LIST_NDIM(8,15,7)) );
    EXPECT_EQ( intVec1-9         , IntVect(LIST_NDIM(-6,1,-7)) );
    EXPECT_EQ( intVec1*2         , IntVect(LIST_NDIM(6,20,4)) );
    EXPECT_EQ( 2*intVec1         , IntVect(LIST_NDIM(6,20,4)) );
    EXPECT_EQ( intVec1/2         , IntVect(LIST_NDIM(1,5,1)) );
    EXPECT_EQ( intVec1.product() , CONCAT_NDIM(3,*10,*2) );

    //test operators for RealVect
    float eps = 1.0e-14;
    for (int i=0;i<NDIM;++i) {
        EXPECT_NEAR(realVec2[i] ,
                RealVect(LIST_NDIM(3.0_wp,10.0_wp,2.0_wp))[i] , eps );
        EXPECT_NEAR((realVec1+realVec2)[i] ,
                RealVect(LIST_NDIM(4.5_wp,13.2_wp,7.8_wp))[i] , eps);
        EXPECT_NEAR((realVec1-realVec2)[i] ,
                RealVect(LIST_NDIM(-1.5_wp,-6.8_wp,3.8_wp))[i] , eps);
        EXPECT_NEAR((realVec1*realVec2)[i] ,
                RealVect(LIST_NDIM(4.5_wp,32.0_wp,11.6_wp))[i] , eps);
        EXPECT_NEAR((realVec1*-3.14_wp)[i] ,
                RealVect(LIST_NDIM(-4.71_wp,-10.048_wp,-18.212_wp))[i] , eps);
        EXPECT_NEAR((-3.14_wp*realVec1)[i] ,
                RealVect(LIST_NDIM(-4.71_wp,-10.048_wp,-18.212_wp))[i] , eps);
        EXPECT_NEAR((realVec1/2.0_wp)[i] ,
                RealVect(LIST_NDIM(0.75_wp,1.6_wp,2.9_wp))[i] , eps);
    }
}

TEST_F(GridUnitTest,ProbConfigGetters){
    Grid::instance().initDomain(Simulation::setInitialConditions_block,
                                Simulation::errorEstMaximal);
    float eps = 1.0e-14;
    int count;

    Grid& grid = Grid::instance();
    RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
    RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
    IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    IntVect  actual_dhi = nBlocks*nCells;

    // Testing Grid::getMaxRefinement and getMaxLevel
    EXPECT_EQ(grid.getMaxRefinement() , LREFINE_MAX-1);
    EXPECT_EQ(grid.getMaxLevel()      , LREFINE_MAX-1);

    // Testing Grid::getProb{Lo,Hi}
    RealVect probLo   = grid.getProbLo();
    RealVect probHi   = grid.getProbHi();
    for (int i=0;i<NDIM;++i) {
        EXPECT_NEAR(probLo[i] , actual_min[i] , eps);
        EXPECT_NEAR(probHi[i] , actual_max[i] , eps);
    }

    // Testing Grid::getDomain{Lo,Hi} and getDeltas
    for (int lev=0; lev<=grid.getMaxLevel(); ++lev) {
        if(lev>0) {
            actual_dhi = actual_dhi * 2;
            actual_deltas = actual_deltas * 0.5_wp;
        }

        IntVect  domainLo = grid.getDomainLo(lev);
        IntVect  domainHi = grid.getDomainHi(lev);
        EXPECT_EQ( domainLo, IntVect(LIST_NDIM(0,0,0)) );
        EXPECT_EQ( domainHi, actual_dhi - 1 );

        RealVect deltas = grid.getDeltas(lev);
        for(int i=1;i<NDIM;++i) {
            EXPECT_NEAR(actual_deltas[i] , deltas[i], eps);
        }
    }

}

TEST_F(GridUnitTest,PerTileGetters){
    Grid::instance().initDomain(Simulation::setInitialConditions_block,
                                Simulation::errorEstMaximal);
    float eps = 1.0e-14;
    int count;

    Grid& grid = Grid::instance();
    RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
    RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
    IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    Real actual_vol = actual_deltas.product();
    RealVect actual_fa;
    for (int i=0;i<NDIM;++i) {
        int p1 = int(i==0);
        int p2 = 2 - int(i==2);
        actual_fa[i] = CONCAT_NDIM( 1.0_wp, *actual_deltas[p1], *actual_deltas[p2] );
    }

    // Test Tile::getCenterCoords with tile iterator
    count = 0;
    for (int lev = 0; lev<=grid.getMaxLevel(); ++lev) {
        if(lev>0) {
            actual_deltas = actual_deltas * 0.5_wp;
        }

        for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
            count++;
            if(count%3 != 0) continue;

            std::unique_ptr<Tile> tileDesc = ti->buildCurrentTile();
            RealVect sumVec = RealVect(tileDesc->lo()+tileDesc->hi()
                                       - 2*grid.getDomainLo(lev) + 1);
            RealVect coords = actual_min + actual_deltas*sumVec*0.5_wp;

            RealVect blkCenterCoords = tileDesc->getCenterCoords();
            for(int i=1;i<NDIM;++i) {
                ASSERT_NEAR(coords[i] , blkCenterCoords[i], eps);
            }
        }
    }


    // Test Grid::getCellVolume and Grid::getCellFaceAreaLo with cell-by-cell iterator
    /*count = 0;
    for (amrex::MFIter itor(grid.unk(),amrex::IntVect(1)); itor.isValid(); ++itor) {
        count++;
        if(count%7 != 0) continue;
        TileAmrex tileDesc(itor, 0);
        IntVect coord = tileDesc.lo();
        ASSERT_NEAR( actual_vol , grid.getCellVolume(0,coord) , eps);
        for(int i=1;i<NDIM;++i) {
            ASSERT_NEAR( actual_fa[i] , grid.getCellFaceAreaLo(0,i,coord) , eps);
        }
    }*/
}

TEST_F(GridUnitTest,MultiCellGetters){
    Grid::instance().initDomain(Simulation::setInitialConditions_block,
                                Simulation::errorEstMaximal);
    float eps = 1.0e-14;

    Grid& grid = Grid::instance();
    RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
    RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
    IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    Real actual_vol = actual_deltas.product();
    RealVect actual_fa;
    for (int i=0;i<NDIM;++i) {
        int p1 = int(i==0);
        int p2 = 2 - int(i==2);
        actual_fa[i] = CONCAT_NDIM( 1.0_wp, *actual_deltas[p1],
                                    *actual_deltas[p2] );
    }

    for (int lev = 0; lev<=grid.getMaxLevel(); ++lev) {
        if(lev>0) {
            actual_deltas = actual_deltas * 0.5_wp;
            actual_vol    = actual_deltas.product();
            actual_fa     = actual_fa * CONCAT_NDIM(1.0_wp,*0.5_wp,*0.5_wp);
        }

        // Test Grid::fillCellVolumes over whole domain.
        IntVect dlo = grid.getDomainLo(lev);
        IntVect dhi = grid.getDomainHi(lev);
        //FArrayBox vol_domain = FArrayBox::buildScratchArray4D(dlo,dhi,1);
        amrex::Box domainBox{ amrex::IntVect(dlo), amrex::IntVect(dhi) };
        {
        amrex::FArrayBox  vol_domain{domainBox,1};
        Real* vol_domain_ptr = vol_domain.dataPtr();
        grid.fillCellVolumes(lev,dlo,dhi,vol_domain_ptr);
        for (int i=dlo.I(); i<=dhi.I(); ++i) {
        for (int j=dlo.J(); j<=dhi.J(); ++j) {
        for (int k=dlo.K(); k<=dhi.K(); ++k) {
            ASSERT_NEAR( vol_domain({LIST_NDIM(i,j,k)},0) , actual_vol , eps);
        }}}
        }

        // Test Grid::fillCellVolumes over an arbitrary range
        RealVect loPtRel{LIST_NDIM(.1,.3,.6)}, hiPtRel{LIST_NDIM(.2,.35,.675)};
        IntVect vlo = dlo + ( RealVect(dhi-dlo)*loPtRel ).floor();
        IntVect vhi = dlo + ( RealVect(dhi-dlo)*hiPtRel ).floor();
        //IntVect vlo{LIST_NDIM(1,2,1)};
        //IntVect vhi{LIST_NDIM(1,3,1)};
        amrex::Box vol_bx{ amrex::IntVect(vlo), amrex::IntVect(vhi) };
        {
        amrex::FArrayBox vol_fab{vol_bx,1};
        Real* vol_ptr = vol_fab.dataPtr();
        grid.fillCellVolumes(lev,vlo,vhi,vol_ptr);
        for (int i=vlo.I(); i<=vhi.I(); ++i) {
        for (int j=vlo.J(); j<=vhi.J(); ++j) {
        for (int k=vlo.K(); k<=vhi.K(); ++k) {
            ASSERT_NEAR( vol_fab({LIST_NDIM(i,j,k)},0) , actual_vol , eps);
        }}}
        }

        // Test Grid::fillCellAreasLo over an arbitrary range
        {
        amrex::FArrayBox  area_fab{vol_bx,1};
        Real* area_ptr = area_fab.dataPtr();
        for(int n=0;n<NDIM;++n) {
            grid.fillCellFaceAreasLo(n,lev,vlo,vhi,area_ptr);
            for (int i=vlo.I(); i<=vhi.I(); ++i) {
            for (int j=vlo.J(); j<=vhi.J(); ++j) {
            for (int k=vlo.K(); k<=vhi.K(); ++k) {
                ASSERT_NEAR( area_fab({LIST_NDIM(i,j,k)},0), actual_fa[n], eps);
            }}}
        }
        }

        // Test Grid::fillCellCoords over an arbitrary range
        {
        int edge[3] = {Edge::Left, Edge::Right, Edge::Center};
        int nElements;
        Real actual_coord;
        for (int j=0; j<3; ++j) {
            //loop over edge cases
            Real offset;
            switch (edge[j]) {
                case Edge::Left:
                    offset = 0.0_wp;
                    break;
                case Edge::Right:
                    offset = 1.0_wp;
                    break;
                case Edge::Center:
                    offset = 0.5_wp;
                    break;
            }
            for(int n=0;n<NDIM;++n) {
                //loop over axis cases
                nElements = vhi[n] - vlo[n] + 1;
                Real coord_ptr[nElements];
                grid.fillCellCoords(n,edge[j],lev,vlo,vhi,coord_ptr);
                for(int i=0; i<nElements; ++i) {
                    actual_coord = actual_min[n] + (Real(vlo[n]+i)+offset)
                                   * actual_deltas[n];
                    ASSERT_NEAR( coord_ptr[i], actual_coord, eps);
                }
            }
        }
        }
    }

}

TEST_F(GridUnitTest,GCFill){
    Grid& grid = Grid::instance();
    float eps = 1.0e-14;

    grid.initDomain(Simulation::setInitialInteriorTest,
                    Simulation::errorEstMaximal);

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

                    EXPECT_NEAR( expected_val, data(i,j,k,DENS_VAR_C), eps);
                }
            }
        }

    } //iterator loop
    } //level loop
}

TEST_F(GridUnitTest,PlotfileOutput){
    Grid& grid = Grid::instance();
    grid.initDomain(Simulation::setInitialConditions_block,
                    Simulation::errorEstMaximal);

    grid.writePlotfile("test_plt_0000");
}

}
