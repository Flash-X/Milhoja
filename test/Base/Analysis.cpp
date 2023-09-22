#include "Analysis.h"

#include <iomanip>
#include <fstream>
#include <algorithm>

#include <Milhoja_axis.h>
#include <Milhoja_edge.h>
#include <Milhoja_FArray4D.h>
#include <Milhoja_Grid.h>
#include <Milhoja_TileWrapper.h>

#include "Base.h"

using namespace milhoja;

namespace Analysis {
    double                energyScaleFactor;
    unsigned int          nCells;
    std::vector<double>   L_inf_dens;
    std::vector<double>   meanAbsError_dens;
    std::vector<double>   L_inf_ener;
    std::vector<double>   meanAbsError_ener;
}

void Analysis::initialize(const unsigned int nBlocks) {
    nCells = 0;
    L_inf_dens.assign(nBlocks, 0.0);
    L_inf_ener.assign(nBlocks, 0.0);
    meanAbsError_dens.assign(nBlocks, 0.0);
    meanAbsError_ener.assign(nBlocks, 0.0);

    // Pretend that we load this from the RuntimeParameter unit
    energyScaleFactor = 5.0;
}

void   Analysis::computeErrors(const IntVect& lo, const IntVect& hi,
                               const FArray1D& xCoords, const FArray1D& yCoords,
                               const FArray4D& U, const int idx) {
    Real    x          = 0.0;
    Real    y          = 0.0;
    Real    absErr     = 0.0;
    Real    maxAbsErr1 = 0.0;
    Real    maxAbsErr2 = 0.0;
    double  UExpected  = 0.0;
    nCells = 0;
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            y = yCoords(j);
            for (int i = lo.I(); i <= hi.I(); ++i) {
                x = xCoords(i);

                UExpected = (18.0*x - 12.0*y - 1.0);
                absErr = fabs(UExpected - U(i, j, k, DENS_VAR));
                if (absErr > maxAbsErr1) {
                     maxAbsErr1 = absErr;
                }

                UExpected = (  48.0*x*x - 18.0*x
                             - 12.0*y*y + 12.0*y
                             - 2.0);
                absErr = fabs(UExpected - U(i, j, k, ENER_VAR));
                if (absErr > maxAbsErr2) {
                     maxAbsErr2 = absErr;
                }

                ++nCells;
            }
        }
    }

    L_inf_dens[idx] = maxAbsErr1;
    L_inf_ener[idx] = maxAbsErr2;
}

void Analysis::densityErrors(double* L_inf, double* meanAbsError) {
    *L_inf = *std::max_element(std::begin(L_inf_dens), std::end(L_inf_dens));
    *meanAbsError = 0.0;
}

void Analysis::energyErrors(double* L_inf, double* meanAbsError) {
    *L_inf = *std::max_element(std::begin(L_inf_ener), std::end(L_inf_ener));
    *meanAbsError = 0.0;
}

void Analysis::writeToFile(const std::string& filename) {
    // TODO: Don't hardcode level
    RealVect  deltas = Grid::instance().getDeltas(0);
    Real      dx = deltas[Axis::I];
    Real      dy = deltas[Axis::J];

    double Linf_d = *std::max_element(std::begin(L_inf_dens), std::end(L_inf_dens));
    double Linf_e = *std::max_element(std::begin(L_inf_ener), std::end(L_inf_ener));

    std::ofstream   fptr;
    fptr.open(filename, std::ios::out);
    fptr << "#dx,dy,Linf Density,Linf Energy\n";
    fptr << std::setprecision(15) << dx << "," << dy << ",";
    fptr << std::setprecision(15) << Linf_d << "," << Linf_e << std::endl;
    fptr.close();
}

