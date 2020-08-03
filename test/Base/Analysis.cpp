#include "Analysis.h"

#include <iomanip>
#include <fstream>
#include <algorithm>

#include "Grid_Axis.h"
#include "Grid_Edge.h"
#include "Grid_IntTriple.h"
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

    const IntVect   lo = tileDesc->lo();
    const IntVect   hi = tileDesc->hi();
    const FArray4D  f  = tileDesc->data();

    Real    xCoords[hi[Axis::I] - lo[Axis::I] + 1];
    Real    yCoords[hi[Axis::J] - lo[Axis::J] + 1];
    grid.fillCellCoords(Axis::I, Edge::Center, tileDesc->level(),
                        lo, hi, xCoords); 
    grid.fillCellCoords(Axis::J, Edge::Center, tileDesc->level(),
                        lo, hi, yCoords); 

    const IntTriple lo3 = lo.asTriple();
    const IntTriple hi3 = hi.asTriple();
    Real    x            = 0.0;
    Real    y            = 0.0;
    int     i0           = lo3[Axis::I];
    int     j0           = lo3[Axis::J];
    int     k0           = lo3[Axis::K];
    Real    absErr       = 0.0;
    Real    maxAbsErr1   = 0.0;
    Real    sum1         = 0.0;
    Real    maxAbsErr2   = 0.0;
    Real    sum2         = 0.0;
    double  fExpected    = 0.0;
    nCells = 0;
    for         (int k = lo3[Axis::K]; k <= hi3[Axis::K]; ++k) {
        for     (int j = lo3[Axis::J]; j <= hi3[Axis::J]; ++j) {
            y = yCoords[j-j0];
            for (int i = lo3[Axis::I]; i <= hi3[Axis::I]; ++i) {
                x = xCoords[i-i0];

                fExpected = (18.0*x - 12.0*y - 1.0);
                absErr = fabs(fExpected - f(i, j, k, DENS_VAR_C));
                sum1 += absErr;
                if (absErr > maxAbsErr1) {
                     maxAbsErr1 = absErr;
                }
 
                fExpected = energyScaleFactor*x*y*(  48.0*x*x - 18.0*x
                                                   - 12.0*y*y + 12.0*y
                                                   - 2.0); 
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

