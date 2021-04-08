#include "Simulation.h"

#include "Grid.h"
#include "Tile.h"
#include "Grid_Axis.h"
#include "Grid_Edge.h"

#include <cmath>
#include <algorithm>
#include <iostream>

#include "Grid_REAL.h"
#include "FArray2D.h"

#include "constants.h"
#include "Flash.h"
#include "Flash_par.h"

/**
  *
  */
void Simulation::setInitialConditions_tile_cpu(const int tId,
                                               orchestration::DataItem* dataItem) {
    using namespace orchestration;

    Tile*  tileDesc = dynamic_cast<Tile*>(dataItem);

    const unsigned int  level    = tileDesc->level();
    const IntVect       loGC     = tileDesc->loGC();
    const IntVect       hiGC     = tileDesc->hiGC();
    FArray4D            solnData = tileDesc->data();
    const RealVect      deltas   = tileDesc->deltas();

    Grid&   grid = Grid::instance();
    FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, level,
                                          loGC, hiGC); 
    FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, level,
                                          loGC, hiGC); 
    FArray1D zCoords = grid.getCellCoords(Axis::K, Edge::Center, level,
                                          loGC, hiGC); 

    // Since we potentially have access to the analytical expression of the ICs,
    // why not use this to set GC data rather than rely on GC fill.
    sim::setInitialConditions(loGC, hiGC, level,
                              xCoords, yCoords, zCoords,
                              deltas,
                              solnData);
}

/**
  *
  */
void  sim::setInitialConditions(const orchestration::IntVect& lo,
                                const orchestration::IntVect& hi,
                                const unsigned int level,
                                const orchestration::FArray1D& xCoords,
                                const orchestration::FArray1D& yCoords,
                                const orchestration::FArray1D& zCoords,
                                const orchestration::RealVect& deltas,
                                orchestration::FArray4D& solnData) {


   for   (int k=lo.K(); k<=hi.K(); ++k) {
    for  (int j=lo.J(); j<=hi.J(); ++j) {
     for (int i=lo.I(); i<=hi.I(); ++i) {

         solnData(i,j,k,RHST_VAR_C) = 0.0;
         solnData(i,j,k,TEMP_VAR_C) = 0.0;

      }
     }
   }

}

