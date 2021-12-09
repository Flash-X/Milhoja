#include "Analysis.h"

#include <iomanip>
#include <fstream>
#include <algorithm>

#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "FArray4D.h"
#include "Grid.h"

#include "Base.h"

using namespace orchestration;

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
    Real    sum1       = 0.0;
    Real    maxAbsErr2 = 0.0;
    Real    sum2       = 0.0;
    double  UExpected  = 0.0;
    nCells = 0;
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            y = yCoords(j);
            for (int i = lo.I(); i <= hi.I(); ++i) {
                x = xCoords(i);

                UExpected = (18.0*x - 12.0*y - 1.0);
                absErr = fabs(UExpected - U(i, j, k, DENS_VAR));
                sum1 += absErr;
                if (absErr > maxAbsErr1) {
                     maxAbsErr1 = absErr;
                }

                UExpected = (  48.0*x*x - 18.0*x
                             - 12.0*y*y + 12.0*y
                             - 2.0);
                absErr = fabs(UExpected - U(i, j, k, ENER_VAR));
                sum2 += absErr;
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

void   ActionRoutines::computeErrors_tile_cpu(const int tId, DataItem* dataItem) {
    Tile* tileDesc = dynamic_cast<Tile*>(dataItem);

    Grid&   grid = Grid::instance();

    const int           idx   = tileDesc->gridIndex();
    const unsigned int  level = tileDesc->level();
    const IntVect       lo    = tileDesc->lo();
    const IntVect       hi    = tileDesc->hi();
    const FArray4D      U     = tileDesc->data();

    FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, level,
                                          lo, hi); 
    FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, level,
                                          lo, hi); 

    Analysis::computeErrors(lo, hi, xCoords, yCoords, U, idx);
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

