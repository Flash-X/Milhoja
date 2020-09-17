#include "Analysis.h"

#include <iomanip>
#include <fstream>
#include <algorithm>

#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "FArray4D.h"
#include "Grid.h"
#include "DataPacket.h"

#include "Flash.h"

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
    
void   Analysis::computeErrors_block(const int tId, void* dataItem) {
    Tile* tileDesc = static_cast<Tile*>(dataItem);

    Grid&   grid = Grid::instance();

    const unsigned int  level = tileDesc->level();
    const IntVect       lo    = tileDesc->lo();
    const IntVect       hi    = tileDesc->hi();
    const FArray4D      f     = tileDesc->data();

    FArray1D xCoords = grid.getCellCoords(Axis::I, Edge::Center, level,
                                          lo, hi); 
    FArray1D yCoords = grid.getCellCoords(Axis::J, Edge::Center, level,
                                          lo, hi); 

    Real    x            = 0.0;
    Real    y            = 0.0;
    Real    absErr       = 0.0;
    Real    maxAbsErr1   = 0.0;
    Real    sum1         = 0.0;
    Real    maxAbsErr2   = 0.0;
    Real    sum2         = 0.0;
    double  fExpected    = 0.0;
    nCells = 0;
    for         (int k = lo.K(); k <= hi.K(); ++k) {
        for     (int j = lo.J(); j <= hi.J(); ++j) {
            y = yCoords(j);
            for (int i = lo.I(); i <= hi.I(); ++i) {
                x = xCoords(i);

                fExpected = (18.0*x - 12.0*y - 1.0);
                absErr = fabs(fExpected - f(i, j, k, DENS_VAR_C));
                sum1 += absErr;
                if (absErr > maxAbsErr1) {
                     maxAbsErr1 = absErr;
                }

                fExpected =   4.0*x*x*x*x - 3.0*x*x*x + 2.0*x*x -     x
                            -     y*y*y*y + 2.0*y*y*y - 3.0*y*y + 4.0*y 
                            + 1.0;
                absErr = fabs(fExpected - f(i, j, k, ENER_VAR_C));
                sum2 += absErr;
                if (absErr > maxAbsErr2) {
                     maxAbsErr2 = absErr;
                }
    
                ++nCells;
            }
        }
    }

    L_inf_dens[tileDesc->gridIndex()] = maxAbsErr1;
    L_inf_ener[tileDesc->gridIndex()] = maxAbsErr2;
}

void   Analysis::computeErrors_packet(const int tId, void* dataItem) {
    DataPacket*  packet = static_cast<DataPacket*>(dataItem);

    for (unsigned int i=0; i<packet->nSubItems(); ++i) {
        computeErrors_block(tId, packet->getSubItem(i));
    }
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

