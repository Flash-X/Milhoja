#include "Flash.h"
#include "constants.h"

#include "Grid.h"
#include "Grid_Macros.h"
#include "Grid_Edge.h"
#include "Grid_Axis.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"
#include <AMReX.H>
#include "TileAmrex.h"

// Macro for iterating over all coordinates in the
// region defined by two IntVects lo and hi.
// Middle three arguments are the iteration variables,
// which can be used in 'function'.

#define ITERATE_REGION(lo,hi,i,j,k, function) {\
std::vector<int> lo_vec3 = lo.as3D(); \
std::vector<int> hi_vec3 = hi.as3D(); \
for(int i=lo_vec3[0];i<=hi_vec3[0];++i) {\
for(int j=lo_vec3[1];j<=hi_vec3[1];++j) {\
for(int k=lo_vec3[2];k<=hi_vec3[2];++k) {\
    function \
}}}}


using namespace orchestration;

namespace {

//test fixture
class GridUnitTest : public testing::Test {
protected:
    GridUnitTest(void) {
            Grid::instance().initDomain(Simulation::setInitialConditions_block);
    }

    ~GridUnitTest(void) {
            Grid::instance().destroyDomain();
    }
};

TEST_F(GridUnitTest,VectorClasses){
    using namespace orchestration;
    //test creation and conversion
    IntVect intVec1{LIST_NDIM(3,10,2)};
    RealVect realVec1{LIST_NDIM(1.5_wp,3.2_wp,5.8_wp)};
    IntVect intVec2 = realVec1.floor();
    RealVect realVec2 = RealVect(intVec1);

    //test operators for IntVect
    EXPECT_TRUE( realVec1.round() == IntVect(LIST_NDIM(2,3,6)) );
    EXPECT_TRUE( realVec1.floor() == IntVect(LIST_NDIM(1,3,5)) );
    EXPECT_TRUE( realVec1.ceil() == IntVect(LIST_NDIM(2,4,6)) );
    EXPECT_TRUE( intVec1 != intVec2 );
    EXPECT_TRUE( intVec1+intVec2 == IntVect(LIST_NDIM(4,13,7)) );
    EXPECT_TRUE( intVec1-intVec2 == IntVect(LIST_NDIM(2,7,-3)) );
    EXPECT_TRUE( intVec1*intVec2 == IntVect(LIST_NDIM(3,30,10)) );
    EXPECT_TRUE( intVec1+5 == IntVect(LIST_NDIM(8,15,7)) );
    EXPECT_TRUE( intVec1-9 == IntVect(LIST_NDIM(-6,1,-7)) );
    EXPECT_TRUE( intVec1*2 == IntVect(LIST_NDIM(6,20,4)) );
    EXPECT_TRUE( 2*intVec1 == IntVect(LIST_NDIM(6,20,4)) );
    EXPECT_TRUE( intVec1/2 == IntVect(LIST_NDIM(1,5,1)) );
    EXPECT_TRUE( intVec1.product() == CONCAT_NDIM(3,*10,*2) );

    //test operators for RealVect
    float eps = 1.0e-14;
    for (int i=0;i<NDIM;++i) {
        EXPECT_NEAR( realVec2[i] , RealVect(LIST_NDIM(3.0_wp,10.0_wp,2.0_wp))[i] , eps );
        EXPECT_NEAR( (realVec1+realVec2)[i] , RealVect(LIST_NDIM(4.5_wp,13.2_wp,7.8_wp))[i] , eps);
        EXPECT_NEAR( (realVec1-realVec2)[i] , RealVect(LIST_NDIM(-1.5_wp,-6.8_wp,3.8_wp))[i] , eps);
        EXPECT_NEAR( (realVec1*realVec2)[i] , RealVect(LIST_NDIM(4.5_wp,32.0_wp,11.6_wp))[i] , eps);
        EXPECT_NEAR( (realVec1*-3.14_wp)[i] , RealVect(LIST_NDIM(-4.71_wp,-10.048_wp,-18.212_wp))[i] , eps);
        EXPECT_NEAR( (-3.14_wp*realVec1)[i] , RealVect(LIST_NDIM(-4.71_wp,-10.048_wp,-18.212_wp))[i] , eps);
        EXPECT_NEAR( (realVec1/2.0_wp)[i] , RealVect(LIST_NDIM(0.75_wp,1.6_wp,2.9_wp))[i] , eps);
    }
}

TEST_F(GridUnitTest,ProbConfigGetters){
    float eps = 1.0e-14;
    int count;

    Grid& grid = Grid::instance();
    RealVect actual_min{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
    RealVect actual_max{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
    IntVect nBlocks{LIST_NDIM(N_BLOCKS_X, N_BLOCKS_Y, N_BLOCKS_Z)};
    IntVect nCells{LIST_NDIM(NXB, NYB, NZB)};
    RealVect actual_deltas = (actual_max-actual_min) / RealVect(nBlocks*nCells);

    //EXPECT_TRUE( nBlocks.product() == grid.unk().boxArray().size() );
    //EXPECT_TRUE((nBlocks*nCells).product() == grid.unk().boxArray().numPts());
    //for (unsigned int i=0; i<nBlocks.product(); ++i) {
    //    ASSERT_TRUE(IntVect(grid.unk().boxArray()[i].size()) == nCells);
    //}

    // Testing Grid::getDomain{Lo,Hi}
    RealVect domainLo = grid.getProbLo();
    RealVect domainHi = grid.getProbHi();
    for (int i=0;i<NDIM;++i) {
        EXPECT_NEAR(domainLo[i] , actual_min[i] , eps);
        EXPECT_NEAR(domainHi[i] , actual_max[i] , eps);
    }

    // Testing Grid::getDeltas
    //TODO: loop over all levels when AMR is implemented
    RealVect deltas = grid.getDeltas(0);
    for(int i=1;i<NDIM;++i) {
        EXPECT_NEAR(actual_deltas[i] , deltas[i], eps);
    }

    //Testing Grid::getMaxRefinement and getMaxLevel
    EXPECT_TRUE(0 == grid.getMaxRefinement());
    EXPECT_TRUE(0 == grid.getMaxLevel());
}

TEST_F(GridUnitTest,PerTileGetters){
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

    // Test Grid::getBlkCenterCoords with tile iterator
    count = 0;
    for (TileIter ti = grid.buildTileIter(0); ti.isValid(); ++ti) {
        count++;
        if(count%3 != 0) continue;

        std::unique_ptr<Tile> tileDesc = ti.buildCurrentTile();
        RealVect sumVec = RealVect(tileDesc->lo()+tileDesc->hi()+1);
        RealVect coords = actual_min + actual_deltas*sumVec*0.5_wp;

        RealVect blkCenterCoords = grid.getBlkCenterCoords(*tileDesc);
        for(int i=1;i<NDIM;++i) {
            ASSERT_NEAR(coords[i] , blkCenterCoords[i], eps);
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
        actual_fa[i] = CONCAT_NDIM( 1.0_wp, *actual_deltas[p1], *actual_deltas[p2] );
    }

    // Test Grid::fillCellVolumes over whole domain.
    IntVect dlo = IntVect( LIST_NDIM(0,0,0) );
    IntVect dhi = nBlocks*nCells-1;
    amrex::Box domainBox{ amrex::IntVect(dlo), amrex::IntVect(dhi) };

    amrex::FArrayBox  vol_domain{domainBox,1};
    Real* vol_domain_ptr = vol_domain.dataPtr();
    grid.fillCellVolumes(0,dlo,dhi,vol_domain_ptr);
    ITERATE_REGION(dlo,dhi,i,j,k,
        EXPECT_NEAR( vol_domain({LIST_NDIM(i,j,k)},0) , actual_vol , eps);
    )

    // Test Grid::fillCellVolumes over an arbitrary range
    IntVect vlo = dlo + ( RealVect(dhi-dlo)*RealVect(LIST_NDIM(.1,.3,.6)) ).floor();
    IntVect vhi = dlo + ( RealVect(dhi-dlo)*RealVect(LIST_NDIM(.2,.35,.675)) ).floor();
    //IntVect vlo{LIST_NDIM(1,2,1)};
    //IntVect vhi{LIST_NDIM(1,3,1)};
    amrex::Box vol_bx{ amrex::IntVect(vlo), amrex::IntVect(vhi) };

    amrex::FArrayBox vol_fab{vol_bx,1};
    Real* vol_ptr = vol_fab.dataPtr();
    grid.fillCellVolumes(0,vlo,vhi,vol_ptr);
    ITERATE_REGION(vlo,vhi,i,j,k,
        EXPECT_NEAR( vol_fab({LIST_NDIM(i,j,k)},0) , actual_vol , eps);
    )

    // Test Grid::fillCellAreasLo over an arbitrary range
    amrex::FArrayBox  area_fab{vol_bx,1};
    Real* area_ptr = area_fab.dataPtr();
    for(int n=0;n<NDIM;++n) {
        grid.fillCellFaceAreasLo(n,0,vlo,vhi,area_ptr);
        ITERATE_REGION(vlo,vhi,i,j,k,
            EXPECT_NEAR( area_fab({LIST_NDIM(i,j,k)},0), actual_fa[n], eps);
        )
    }

    // Test Grid::fillCellCoords over an arbitrary range
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
            grid.fillCellCoords(n,edge[j],0,vlo,vhi,coord_ptr);
            for(int i=0; i<nElements; ++i) {
                actual_coord = actual_min[n] + (Real(vlo[n]+i)+offset) * actual_deltas[n];
                EXPECT_NEAR( coord_ptr[i], actual_coord, eps);
            }
        }
    }

}

}
