#include "Flash.h"
#include "constants.h"
#include "Flash_par.h"

#include "OrchestrationLogger.h"
#include "Grid.h"
#include "Grid_Macros.h"
#include "Grid_Edge.h"
#include "Grid_Axis.h"
#include "setInitialConditions.h"
#include "setInitialInteriorTest.h"
#include "errorEstBlank.h"
#include "errorEstMaximal.h"
#include "errorEstMultiple.h"
#include "gtest/gtest.h"
#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include "Tile.h"

#include <iostream>


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
    Grid::instance().initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                                rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                                rp_Simulation::N_THREADS_FOR_IC,
                                Simulation::errorEstBlank);

    //test creation and conversion
    IntVect intVec1{LIST_NDIM(3,10,2)};
    const IntVect intVecConst{LIST_NDIM(3,10,2)};
    RealVect realVec1{LIST_NDIM(1.5_wp,3.2_wp,5.8_wp)};
    const RealVect realVecConst{LIST_NDIM(1.5_wp,3.2_wp,5.8_wp)};
    IntVect intVec2 = realVec1.floor();
    RealVect realVec2 = RealVect(intVec1);

    //test operators for IntVect
    EXPECT_EQ( intVec1           , intVecConst);
    EXPECT_EQ( realVec1.round()  , IntVect(LIST_NDIM(2,3,6)) );
    EXPECT_EQ( realVec1.floor()  , IntVect(LIST_NDIM(1,3,5)) );
    EXPECT_EQ( realVec1.ceil()   , IntVect(LIST_NDIM(2,4,6)) );
    EXPECT_NE( intVec1           , intVec2 );
    EXPECT_EQ( intVec1+intVec2   , IntVect(LIST_NDIM(4,13,7)) );
    EXPECT_EQ( intVec1-intVec2   , IntVect(LIST_NDIM(2,7,-3)) );
    EXPECT_EQ( intVec1*intVec2   , IntVect(LIST_NDIM(3,30,10)) );
    EXPECT_EQ( intVec1+5         , IntVect(LIST_NDIM(8,15,7)) );
    EXPECT_EQ( 7+intVec1         , IntVect(LIST_NDIM(10,17,9)) );
    EXPECT_EQ( intVec1-9         , IntVect(LIST_NDIM(-6,1,-7)) );
    EXPECT_EQ( intVec1*2         , IntVect(LIST_NDIM(6,20,4)) );
    EXPECT_EQ( 2*intVec1         , IntVect(LIST_NDIM(6,20,4)) );
    EXPECT_EQ( intVec1/2         , IntVect(LIST_NDIM(1,5,1)) );
    EXPECT_EQ( intVec1.product() , CONCAT_NDIM(3,*10,*2) );

    //test operators for RealVect
    float eps = 1.0e-14;
    for (int i=0;i<NDIM;++i) {
        EXPECT_NEAR(realVec1[i] , realVecConst[i] , eps);
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

    //Test logic errors
    int caughtErrors = 0;
    try {
        Real r = realVec1[3];
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        Real r = realVecConst[3];
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        int x = intVec1[3];
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        int x = intVecConst[3];
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    EXPECT_EQ( caughtErrors, 4);

    // Test output
    std::cout << "Sample IntVect: " << intVec1 << std::endl;
    std::cout << "Sample RealVect: " << realVec1 << std::endl;
}

TEST_F(GridUnitTest,FArrayClasses){
    //Grid::instance().initDomain(ActionRoutines::setInitialConditions_tile_cpu,
    //                            rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
    //                            rp_Simulation::N_THREADS_FOR_IC,
    //                            Simulation::errorEstBlank);

    //test creation and assignment
    Real arrayData[10];
    FArray1D data{arrayData, 0};
    FArray1D scratchData = FArray1D::buildScratchArray1D(5,14);
    IntVect lo{LIST_NDIM(0,0,0)}, hi{LIST_NDIM(2,4,3)};
    Real arrayData4[hi.product() * 3];
    FArray4D data4{arrayData4, lo, hi, 3};
    FArray4D scratchData4 = FArray4D::buildScratchArray4D(lo, hi, 3);

    // Test assignment of 1D
    for (int i=0; i<10; ++i) {
        data(i) = i*5.0_wp;
        scratchData(i+5) = i*7.0_wp;
    }
    float eps = 1.0e-14;
    for (int i=0;i<10;++i) {
        EXPECT_NEAR( data(i), i*5.0_wp, eps);
        EXPECT_NEAR( scratchData(i+5), i*7.0_wp, eps);
    }

    // TODO test assignment of 4D

    //Test logic errors
    int caughtErrors = 0;
    try {
        FArray1D dataNull{ nullptr, 0};
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        FArray4D dataNull{ nullptr, lo, hi, 3};
    } catch (const std::invalid_argument& e) {
        caughtErrors++;
    }
    try {
        FArray4D dataNull{ arrayData4, lo, hi, 0};
    } catch (const std::invalid_argument& e) {
        caughtErrors++;
    }
    try {
        FArray4D dataNull{ arrayData4, hi, lo, 3};
    } catch (const std::invalid_argument& e) {
        caughtErrors++;
    }
    //try {
    //    Real r = data(10); 
    //} catch (const std::logic_error& e) {
    //    caughtErrors++;
    //}
    //try {
    //    Real r = scratchData(4);
    //} catch (const std::logic_error& e) {
    //    caughtErrors++;
    //}
    EXPECT_EQ( caughtErrors, 4);
}

TEST_F(GridUnitTest,ProbConfigGetters){
    Grid::instance().initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                                rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                                rp_Simulation::N_THREADS_FOR_IC,
                                Simulation::errorEstMaximal);
    float eps = 1.0e-14;

    Grid& grid = Grid::instance();
    RealVect actual_min{LIST_NDIM(rp_Grid::X_MIN,
                                  rp_Grid::Y_MIN,
                                  rp_Grid::Z_MIN)};
    RealVect actual_max{LIST_NDIM(rp_Grid::X_MAX,
                                  rp_Grid::Y_MAX,
                                  rp_Grid::Z_MAX)};
    IntVect nBlocks{LIST_NDIM(rp_Grid::N_BLOCKS_X,
                              rp_Grid::N_BLOCKS_Y,
                              rp_Grid::N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    IntVect  actual_dhi = nBlocks*nCells;

    // Test getNumberLocalBlocks
    unsigned int nBlocksLocal   = grid.getNumberLocalBlocks();
    EXPECT_EQ(nBlocksLocal, nBlocks.product() );

    // Testing Grid::getMaxRefinement and getMaxLevel
    EXPECT_EQ(grid.getMaxRefinement() , rp_Grid::LREFINE_MAX-1);
    EXPECT_EQ(grid.getMaxLevel()      , rp_Grid::LREFINE_MAX-1);

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
        for(int i=0;i<NDIM;++i) {
            EXPECT_NEAR(actual_deltas[i] , deltas[i], eps);
        }
    }

    // Test subcellGeometry
    Real subVols[32];
    grid.subcellGeometry(8,4,2,12.8,subVols);
    for (int i=0;i<32;++i) {
        EXPECT_NEAR( subVols[i], 0.2, eps);
    }

}

TEST_F(GridUnitTest,PerTileGetters){
    Grid::instance().initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                                rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                                rp_Simulation::N_THREADS_FOR_IC,
                                Simulation::errorEstMaximal);
    float eps = 1.0e-14;
    int count;

    Grid& grid = Grid::instance();
    RealVect actual_min{LIST_NDIM(rp_Grid::X_MIN,
                                  rp_Grid::Y_MIN,
                                  rp_Grid::Z_MIN)};
    RealVect actual_max{LIST_NDIM(rp_Grid::X_MAX,
                                  rp_Grid::Y_MAX,
                                  rp_Grid::Z_MAX)};
    IntVect nBlocks{LIST_NDIM(rp_Grid::N_BLOCKS_X,
                              rp_Grid::N_BLOCKS_Y,
                              rp_Grid::N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);

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
            for(int i=0;i<NDIM;++i) {
                ASSERT_NEAR(coords[i] , blkCenterCoords[i], eps);
            }
        }
    }


    // Test Grid::getCellVolume and Grid::getCellFaceAreaLo on lo vertices
    count = 0;
    actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    Real actual_vol;
    RealVect actual_fa;
    for (int lev = 0; lev<=grid.getMaxLevel(); ++lev) {
        if(lev>0) {
            actual_deltas = actual_deltas * 0.5_wp;
        }
        actual_vol = actual_deltas.product();
        for (int i=0;i<NDIM;++i) {
            actual_fa[i] = actual_vol / actual_deltas[i];
        }
#if NDIM==1
        actual_fa[0] = 0.0_wp;
#endif

        for (auto ti = grid.buildTileIter(lev); ti->isValid(); ti->next()) {
            count = (count+1)%7;
            if(count == 0) continue;

            std::unique_ptr<Tile> tileDesc = ti->buildCurrentTile();
            IntVect vert = tileDesc->lo();
            ASSERT_NEAR( actual_vol, grid.getCellVolume(lev,vert), eps);
            for(int i=0;i<NDIM;++i) {
                ASSERT_NEAR( actual_fa[i],
                             grid.getCellFaceAreaLo(i,lev,vert) , eps);
            }
        }
    }
}

TEST_F(GridUnitTest,MultiCellGetters){
    Grid::instance().initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                                rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                                rp_Simulation::N_THREADS_FOR_IC,
                                Simulation::errorEstMaximal);
    float eps = 1.0e-14;

    Grid& grid = Grid::instance();
    RealVect actual_min{LIST_NDIM(rp_Grid::X_MIN,
                                  rp_Grid::Y_MIN,
                                  rp_Grid::Z_MIN)};
    RealVect actual_max{LIST_NDIM(rp_Grid::X_MAX,
                                  rp_Grid::Y_MAX,
                                  rp_Grid::Z_MAX)};
    IntVect nBlocks{LIST_NDIM(rp_Grid::N_BLOCKS_X,
                              rp_Grid::N_BLOCKS_Y,
                              rp_Grid::N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    Real actual_vol = actual_deltas.product();
    RealVect actual_fa;
    for (int i=0;i<NDIM;++i) {
        actual_fa[i] = actual_vol / actual_deltas[i];
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
            ASSERT_NEAR( vol_domain(amrex::IntVect(LIST_NDIM(i,j,k)),0) , actual_vol , eps);
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
            ASSERT_NEAR( vol_fab(amrex::IntVect(LIST_NDIM(i,j,k)),0) , actual_vol , eps);
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
                ASSERT_NEAR( area_fab(amrex::IntVect(LIST_NDIM(i,j,k)),0), actual_fa[n], eps);
            }}}
        }
        }

        // Test Grid::getCellCoords over an arbitrary range
        {
        int edge[3] = {Edge::Left, Edge::Right, Edge::Center};
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
            for(int n=0;n<MDIM;++n) {
                //loop over axis cases
                FArray1D coord_ptr = grid.getCellCoords(n,edge[j],lev,vlo,vhi);
                if (n<NDIM) {
                    for(int i=vlo[n]; i<=vhi[n]; ++i) {
                        actual_coord = actual_min[n] + (Real(i)+offset)
                                       * actual_deltas[n];
                        ASSERT_NEAR( coord_ptr(i), actual_coord, eps);
                    }
                } else {
                    // Test default value for axes above NDIM
                    ASSERT_NEAR( coord_ptr(0), 0.0_wp, eps);
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
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
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

TEST_F(GridUnitTest,MultipleLevels){
    Grid& grid = Grid::instance();
    float eps = 1.0e-10;
    grid.initDomain(Simulation::setInitialInteriorTest,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstMultiple);

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
    grid.regrid();

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
    grid.regrid();

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

TEST_F(GridUnitTest,LogicErrors){
    Grid& grid = Grid::instance();
    int caughtErrors = 0;

    // Try instantiating Grid & Logger after it's already been done
    try {
        Grid::instantiate();
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        Logger::instantiate("GridUnitTest.log");
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    // Try Logger::setLogFilename with empty name
    try {
        Logger::setLogFilename("");
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }

    // Try initDomain with invalid arguments
    try {
        grid.initDomain(nullptr,
                        rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                        rp_Simulation::N_THREADS_FOR_IC,
                        Simulation::errorEstMaximal);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    //try {
    //    grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu,
    //                    2,
    //                    rp_Simulation::N_THREADS_FOR_IC,
    //                    Simulation::errorEstMaximal);
    //} catch (const std::invalid_argument& e) {
    //    caughtErrors++;
    //}
    //grid.destroyDomain();
    //try {
    //    grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu,
    //                    rp_Simulation::N_THREADS_FOR_IC + 1,
    //                    rp_Simulation::N_THREADS_FOR_IC,
    //                    Simulation::errorEstMaximal);
    //} catch (const std::invalid_argument& e) {
    //    caughtErrors++;
    //}
    //grid.destroyDomain();
    try {
        grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                        rp_Simulation::N_THREADS_FOR_IC,
                        0,
                        Simulation::errorEstMaximal);
    } catch (const std::invalid_argument& e) {
        caughtErrors++;
    }
    grid.destroyDomain();

    grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstMaximal);

    // Try initDomain after it's already been called
    try {
        grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                        rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                        rp_Simulation::N_THREADS_FOR_IC,
                        Simulation::errorEstMaximal);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        grid.buildTileIter(grid.getMaxLevel() + 1);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }

    // Try invalid axis/edge cases
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        int wrongAxis = 3;
        grid.getCellCoords(wrongAxis,0,0,iv,iv);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        int wrongEdge = -1;
        grid.getCellCoords(0,wrongEdge,0,iv,iv);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        int wrongAxis = 3;
        Real* rp;
        grid.fillCellFaceAreasLo(wrongAxis,0,iv,iv,rp);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }

    // Try unimplemented routines
    try {
        grid.Grid::getDeltas(0);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        grid.Grid::getCellFaceAreaLo(0,0,iv);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        grid.Grid::getCellVolume(0,iv);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        grid.Grid::getCellCoords(0,0,0,iv,iv);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        int wrongAxis = 3;
        grid.Grid::getCellCoords(wrongAxis,0,0,iv,iv);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        int wrongEdge = -1;
        grid.Grid::getCellCoords(0,wrongEdge,0,iv,iv);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        Real* rp;
        grid.Grid::fillCellFaceAreasLo(0,0,iv,iv,rp);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        int wrongAxis = 3;
        Real* rp;
        grid.Grid::fillCellFaceAreasLo(wrongAxis,0,iv,iv,rp);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }
    try {
        IntVect iv{LIST_NDIM(0,0,0)};
        Real* rp;
        grid.Grid::fillCellVolumes(0,iv,iv,rp);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }

    EXPECT_EQ( caughtErrors, 19);
}

TEST_F(GridUnitTest,PlotfileOutput){
    Grid& grid = Grid::instance();
    grid.initDomain(ActionRoutines::setInitialConditions_tile_cpu,
                    rp_Simulation::N_DISTRIBUTOR_THREADS_FOR_IC,
                    rp_Simulation::N_THREADS_FOR_IC,
                    Simulation::errorEstMaximal);

    grid.writePlotfile("test_plt_0000");
}

}
