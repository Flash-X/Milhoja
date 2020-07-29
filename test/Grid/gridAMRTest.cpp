#include "Flash.h"
#include "constants.h"

#include "Grid.h"
#include "Grid_AmrCoreFlash.h"
#include "Grid_Macros.h"
#include "Grid_Edge.h"
#include "Grid_Axis.h"
#include "setInitialConditions_block.h"
#include "gtest/gtest.h"
#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrMesh.H>

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
class GridAMRTest : public testing::Test {
protected:
    GridAMRTest(void) {
            RealVect probMin{LIST_NDIM(X_MIN,Y_MIN,Z_MIN)};
            RealVect probMax{LIST_NDIM(X_MAX,Y_MAX,Z_MAX)};
            IntVect  nBlocks{LIST_NDIM(N_BLOCKS_X,N_BLOCKS_Y,N_BLOCKS_Z)};
            Grid&    grid = Grid::instance();
            grid.initDomain(probMin,probMax,nBlocks,NUNKVAR,
                            Simulation::setInitialConditions_block);
    }

    ~GridAMRTest(void) {
            Grid::instance().destroyDomain();
    }
};

TEST_F(GridAMRTest,Trivial){
    EXPECT_TRUE( 1==1 );

}

}
