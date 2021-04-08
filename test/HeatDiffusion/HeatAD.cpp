#include "HeatAD.h"

#include <cmath>

#include "Flash.h"

namespace HeatAD {
  using namespace orchestration;
  orchestration::Real alpha = 1.0;
};

void HeatAD::diffusion(orchestration::FArray4D& solnData,
                       const orchestration::RealVect& deltas,
                       const orchestration::Real diffusion_coeff,
                       const orchestration::IntVect& lo,
                       const orchestration::IntVect& hi) {

   using namespace orchestration;

   for   (int k=lo.K(); k<=hi.K(); ++k) {
    for  (int j=lo.J(); j<=hi.J(); ++j) {
     for (int i=lo.I(); i<=hi.I(); ++i) {

         solnData(i,j,k,RHST_VAR_C) = (diffusion_coeff/pow(deltas.I(),2))*(solnData(i+1, j,   k, TEMP_VAR_C)+\
                                                                           solnData(i-1, j,   k, TEMP_VAR_C)-\
                                                                         2*solnData(i,   j,   k, TEMP_VAR_C))+\
                                      (diffusion_coeff/pow(deltas.J(),2))*(solnData(i,   j+1, k, TEMP_VAR_C)+\
                                                                           solnData(i,   j-1, k, TEMP_VAR_C)-\
                                                                         2*solnData(i,   j,   k, TEMP_VAR_C));

       }
     }
   }

};

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
