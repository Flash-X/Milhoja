#include "HeatAD.h"

#include <cmath>
#include <algorithm>

#include "Flash.h"
#include "Grid_IntVect.h"
#include "Grid_RealVect.h"
#include "FArray1D.h"
#include "FArray4D.h"
#include "DataItem.h"

void HeatAD::solve(orchestration::FArray4D& solnData,
                   const orchestration::Real dt,
                   const orchestration::IntVect& lo,
                   const orchestration::IntVect& hi) {

   using namespace orchestration;

   for   (int k=lo.K(); k<=hi.K(); ++k) {
    for  (int j=lo.J(); j<=hi.J(); ++j) {
     for (int i=lo.I(); i<=hi.I(); ++i) {

         solnData(i,j,k,TEMP_VAR_C) = solnData(i,j,k,TEMP_VAR_C) + dt*solnData(i,j,k,RHST_VAR_C);

       }
     }
   }

};
