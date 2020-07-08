#include "Analysis.h"

#include <iomanip>
#include <fstream>
#include <algorithm>

#include "Flash.h"
#include "Grid.h"
#include "DataPacket.h"

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
    amrex::MultiFab&   unk = grid.unk();
    amrex::Geometry&   geometry = grid.geometry();

    int               gid = tileDesc->gridIndex();
    const amrex::Dim3 lo  = tileDesc->lo();
    const amrex::Dim3 hi  = tileDesc->hi();

    amrex::FArrayBox&   fab = unk[gid];
    amrex::Array4<amrex::Real> const& f = fab.array();

    amrex::Real  x            = 0.0;
    amrex::Real  y            = 0.0;
    amrex::Real  absErr       = 0.0;
    amrex::Real  maxAbsErr1   = 0.0;
    amrex::Real  sum1         = 0.0;
    amrex::Real  maxAbsErr2   = 0.0;
    amrex::Real  sum2         = 0.0;
    double       fExpected    = 0.0;
    nCells = 0;
    for     (int j = lo.y; j <= hi.y; ++j) {
        y = geometry.CellCenter(j, 1);
        for (int i = lo.x; i <= hi.x; ++i) {
            x = geometry.CellCenter(i, 0);

            fExpected = (18.0*x - 12.0*y - 1.0);
            absErr = fabs(fExpected - f(i, j, lo.z, DENS_VAR_C));
            sum1 += absErr;
            if (absErr > maxAbsErr1) {
                 maxAbsErr1 = absErr;
            }
   
            fExpected = energyScaleFactor*x*y*(  48.0*x*x - 18.0*x
                                               - 12.0*y*y + 12.0*y
                                               - 2.0); 
            absErr = fabs(fExpected - f(i, j, lo.z, ENER_VAR_C));
            sum2 += absErr;
            if (absErr > maxAbsErr2) {
                 maxAbsErr2 = absErr;
            }
    
            ++nCells;
        }
    }

    L_inf_dens[gid] = maxAbsErr1;
    L_inf_ener[gid] = maxAbsErr2;
}

void   Analysis::computeErrors_packet(const int tId, void* dataItem) {
    DataPacket*  packet = static_cast<DataPacket*>(dataItem);

    for (unsigned int i=0; i<packet->tileList.size(); ++i) {
        Tile&    work = packet->tileList[i];
        computeErrors_block(tId, &work);
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
    amrex::Geometry geometry = Grid::instance().geometry();
    amrex::Real  dx = geometry.CellSize(0);
    amrex::Real  dy = geometry.CellSize(1);

    double Linf_d = *std::max_element(std::begin(L_inf_dens), std::end(L_inf_dens));
    double Linf_e = *std::max_element(std::begin(L_inf_ener), std::end(L_inf_ener));

    std::ofstream   fptr;
    fptr.open(filename, std::ios::out);
    fptr << "#dx,dy,Linf Density,Linf Energy\n";
    fptr << std::setprecision(15) << dx << "," << dy << ",";
    fptr << std::setprecision(15) << Linf_d << "," << Linf_e << std::endl;
    fptr.close();
}

