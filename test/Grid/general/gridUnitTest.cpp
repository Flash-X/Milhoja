#include <iostream>

#include <AMReX.H>
#include <AMReX_FArrayBox.H>

#include <gtest/gtest.h>

#include <Milhoja.h>
#include <Milhoja_Grid.h>
#include <Milhoja_edge.h>
#include <Milhoja_Tile.h>
#include <Milhoja_ThreadTeamDataType.h>
#include <Milhoja_test.h>

#include "RuntimeParameters.h"
#include "Simulation.h"
#include "setInitialConditions.h"

#include "cpu_tf_ic.h"
#include "Tile_cpu_tf_ic.h"

using namespace milhoja;

namespace {

TEST(GridUnitTest,VectorClasses){
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
    for (int i=0;i<MILHOJA_NDIM;++i) {
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

TEST(GridUnitTest,FArrayClasses){
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

TEST(GridUnitTest,ProbConfigGetters){
    float eps = 1.0e-14;
    int count;

    Grid&               grid = Grid::instance();
    RuntimeParameters&  RPs  = RuntimeParameters::instance();

    Real           xMin{RPs.getReal("Grid", "xMin")};
    Real           xMax{RPs.getReal("Grid", "xMax")};
    Real           yMin{RPs.getReal("Grid", "yMin")};
    Real           yMax{RPs.getReal("Grid", "yMax")};
    Real           zMin{RPs.getReal("Grid", "zMin")};
    Real           zMax{RPs.getReal("Grid", "zMax")};
    int            nBlocksX{RPs.getInt("Grid", "nBlocksX")};
    int            nBlocksY{RPs.getInt("Grid", "nBlocksY")};
    int            nBlocksZ{RPs.getInt("Grid", "nBlocksZ")};
    int            nxb{MILHOJA_TEST_NXB};
    int            nyb{MILHOJA_TEST_NYB};
    int            nzb{MILHOJA_TEST_NZB};
    unsigned int   lRefineMax{RPs.getUnsignedInt("Grid", "finestRefinementLevel")};

    RealVect actual_min{LIST_NDIM(xMin, yMin, zMin)};
    RealVect actual_max{LIST_NDIM(xMax, yMax, zMax)};
    IntVect nBlocks{LIST_NDIM(nBlocksX, nBlocksY, nBlocksZ)};
    IntVect nCells{LIST_NDIM(nxb, nyb, nzb)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    IntVect  actual_dhi = nBlocks*nCells;

    // Testing Grid::getMaxRefinement and getMaxLevel
    EXPECT_EQ(grid.getMaxRefinement(), lRefineMax-1);
    EXPECT_EQ(grid.getMaxLevel()     , lRefineMax-1);

    // Testing Grid::getProb{Lo,Hi}
    RealVect probLo   = grid.getProbLo();
    RealVect probHi   = grid.getProbHi();
    for (int i=0;i<MILHOJA_NDIM;++i) {
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
        for(int i=1;i<MILHOJA_NDIM;++i) {
            EXPECT_NEAR(actual_deltas[i] , deltas[i], eps);
        }
    }

    Real subVols[32];
    grid.subcellGeometry(8,4,2,12.8,subVols);
    for (int i=0;i<32;++i) {
        EXPECT_NEAR( subVols[i], 0.2, eps);
    }

}

TEST(GridUnitTest,PerTileGetters){
    float eps = 1.0e-14;
    int count;

    Grid&               grid = Grid::instance();
    RuntimeParameters&  RPs  = RuntimeParameters::instance();

    Real  xMin{RPs.getReal("Grid", "xMin")};
    Real  xMax{RPs.getReal("Grid", "xMax")};
    Real  yMin{RPs.getReal("Grid", "yMin")};
    Real  yMax{RPs.getReal("Grid", "yMax")};
    Real  zMin{RPs.getReal("Grid", "zMin")};
    Real  zMax{RPs.getReal("Grid", "zMax")};
    int   nBlocksX{RPs.getInt("Grid", "nBlocksX")};
    int   nBlocksY{RPs.getInt("Grid", "nBlocksY")};
    int   nBlocksZ{RPs.getInt("Grid", "nBlocksZ")};
    int   nxb{MILHOJA_TEST_NXB};
    int   nyb{MILHOJA_TEST_NYB};
    int   nzb{MILHOJA_TEST_NZB};

    RealVect actual_min{LIST_NDIM(xMin, yMin, zMin)};
    RealVect actual_max{LIST_NDIM(xMax, yMax, zMax)};
    IntVect nBlocks{LIST_NDIM(nBlocksX, nBlocksY, nBlocksZ)};
    IntVect nCells{LIST_NDIM(nxb, nyb, nzb)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    Real actual_vol = actual_deltas.product();
    RealVect actual_fa;
    for (int i=0;i<MILHOJA_NDIM;++i) {
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
            for(int i=1;i<MILHOJA_NDIM;++i) {
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
        for(int i=1;i<MILHOJA_NDIM;++i) {
            ASSERT_NEAR( actual_fa[i] , grid.getCellFaceAreaLo(0,i,coord) , eps);
        }
    }*/
}

TEST(GridUnitTest,MultiCellGetters){
    float eps = 1.0e-14;

    Grid&               grid = Grid::instance();
    RuntimeParameters&  RPs  = RuntimeParameters::instance();

    Real  xMin{RPs.getReal("Grid", "xMin")};
    Real  xMax{RPs.getReal("Grid", "xMax")};
    Real  yMin{RPs.getReal("Grid", "yMin")};
    Real  yMax{RPs.getReal("Grid", "yMax")};
    Real  zMin{RPs.getReal("Grid", "zMin")};
    Real  zMax{RPs.getReal("Grid", "zMax")};
    int   nBlocksX{RPs.getInt("Grid", "nBlocksX")};
    int   nBlocksY{RPs.getInt("Grid", "nBlocksY")};
    int   nBlocksZ{RPs.getInt("Grid", "nBlocksZ")};
    int   nxb{MILHOJA_TEST_NXB};
    int   nyb{MILHOJA_TEST_NYB};
    int   nzb{MILHOJA_TEST_NZB};

    RealVect actual_min{LIST_NDIM(xMin, yMin, zMin)};
    RealVect actual_max{LIST_NDIM(xMax, yMax, zMax)};
    IntVect nBlocks{LIST_NDIM(nBlocksX, nBlocksY, nBlocksZ)};
    IntVect nCells{LIST_NDIM(nxb, nyb, nzb)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);
    Real actual_vol = actual_deltas.product();
    RealVect actual_fa;
    for (int i=0;i<MILHOJA_NDIM;++i) {
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
        for(int n=0;n<MILHOJA_NDIM;++n) {
            grid.fillCellFaceAreasLo(n,lev,vlo,vhi,area_ptr);
            for (int i=vlo.I(); i<=vhi.I(); ++i) {
            for (int j=vlo.J(); j<=vhi.J(); ++j) {
            for (int k=vlo.K(); k<=vhi.K(); ++k) {
                ASSERT_NEAR( area_fab(amrex::IntVect(LIST_NDIM(i,j,k)),0), actual_fa[n], eps);
            }}}
        }
        }

        // Test Grid::fillCellCoords over an arbitrary range
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
            for(int n=0;n<MILHOJA_NDIM;++n) {
                //loop over axis cases
                FArray1D coord_ptr = grid.getCellCoords(n,edge[j],lev,vlo,vhi);
                for(int i=vlo[n]; i<=vhi[n]; ++i) {
                    actual_coord = actual_min[n] + (Real(i)+offset)
                                   * actual_deltas[n];
                    ASSERT_NEAR( coord_ptr(i), actual_coord, eps);
                }
            }
        }
        }
    }

}

TEST(GridUnitTest,LogicErrors){
    Grid& grid = Grid::instance();
    int caughtErrors = 0;
    try {
        Grid::initialize();
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }

    try {
        grid.initDomain(sim::setInitialConditions_noRuntime);
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }

    try {
        RuntimeAction   initBlock_cpu;
        initBlock_cpu.name = "initBlock_cpu";
        initBlock_cpu.teamType        = ThreadTeamDataType::BLOCK;
        initBlock_cpu.nInitialThreads = 1;
        initBlock_cpu.nTilesPerPacket = 0;
        initBlock_cpu.routine         = cpu_tf_ic::taskFunction;

        Tile_cpu_tf_ic::acquireScratch();
        Tile_cpu_tf_ic    prototype{};
        grid.initDomain(initBlock_cpu, &prototype);
        Tile_cpu_tf_ic::releaseScratch();
    } catch (const std::logic_error& e) {
        caughtErrors++;
    }

    EXPECT_EQ( caughtErrors, 3);
}

TEST(GridUnitTest,PlotfileOutput){
    std::vector<std::string>   names = sim::getVariableNames();

    Grid& grid = Grid::instance();
    grid.writePlotfile("test_plt_0000", names);
}

}

