#include <stdexcept>

#include <AMReX_Vector.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>
#include <AMReX_Box.H>
#include <AMReX_RealBox.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include "constants.h"

template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
Grid<NX,NY,NZ,NGC>*  Grid<NX,NY,NZ,NGC>::instance_ = nullptr;

/**
 * 
 *
 * \return 
 */
template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
Grid<NX,NY,NZ,NGC>*   Grid<NX,NY,NZ,NGC>::instance(void) {
    if (!instance_) {
        instance_ = new Grid();
    }
    return instance_;
}

/**
 * 
 *
 * \return 
 */
template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
Grid<NX,NY,NZ,NGC>::Grid(void) 
    : unk_(nullptr)
{
    amrex::Initialize(MPI_COMM_WORLD);
    destroyDomain();
}

/**
 * 
 */
template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
Grid<NX,NY,NZ,NGC>::~Grid(void) {
    destroyDomain();
    amrex::Finalize();
    instance_ = nullptr;
}

/**
 *
 */
template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
void    Grid<NX,NY,NZ,NGC>::initDomain(const amrex::Real xMin, const amrex::Real xMax,
                                       const amrex::Real yMin, const amrex::Real yMax,
                                       const amrex::Real zMin, const amrex::Real zMax,
                                       const unsigned int nBlocksX,
                                       const unsigned int nBlocksY,
                                       const unsigned int nBlocksZ,
                                       const unsigned int nVars,
                                       SET_IC_FCN initBlock) {
    // TODO: Error check all given parameters
    if (unk_) {
        throw std::logic_error("Grid unit's initDomain already called");
    } else if (!initBlock) {
        throw std::logic_error("Null initBlock function pointer given");
    }

    //***** SETUP DOMAIN, PROBLEM, and MESH
    amrex::IndexType    ccIndexSpace(amrex::IntVect(AMREX_D_DECL(0, 0, 0)));
    amrex::IntVect      domainLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect      domainHi(AMREX_D_DECL(nBlocksX * NX - 1,
                                              nBlocksY * NY - 1,
                                              nBlocksZ * NZ - 1));
    amrex::Box          domain = amrex::Box(domainLo, domainHi, ccIndexSpace);
    amrex::BoxArray     ba(domain);
    ba.maxSize(amrex::IntVect(AMREX_D_DECL(NX, NY, NZ)));
    amrex::DistributionMapping  dm(ba);

    // Setup with Cartesian coordinate and non-periodic BC so that we can set
    // the BC ourselves
    int coordSystem = 0;  // Cartesian
    amrex::RealBox   physicalDomain = amrex::RealBox({AMREX_D_DECL(xMin, yMin, zMin)},
                                                     {AMREX_D_DECL(xMax, yMax, zMax)});
    geometry_ = amrex::Geometry(domain, physicalDomain,
                                coordSystem, {AMREX_D_DECL(0, 0, 0)});

    assert(nBlocksX * nBlocksY * nBlocksZ == ba.size());
    assert(NX*nBlocksX * NY*nBlocksY * NZ*nBlocksZ == ba.numPts());
    for (unsigned int i=0; i<ba.size(); ++i) {
        assert(ba[i].size() == amrex::IntVect(AMREX_D_DECL(NX, NY, NZ)));
    }

    unk_ = new amrex::MultiFab(ba, dm, nVars, NGC);
    for (amrex::MFIter  itor(*unk_); itor.isValid(); ++itor) {
        Tile   tileDesc(itor);
        initBlock(&tileDesc);
    }
}

/**
 *
 */
template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
void    Grid<NX,NY,NZ,NGC>::destroyDomain(void) {
    if (unk_) {
        delete unk_;
        unk_ = nullptr;
    }
    geometry_ = amrex::Geometry();
}

/**
 *
 */
template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
void    Grid<NX,NY,NZ,NGC>::writeToFile(const std::string& filename) const {
    amrex::Vector<std::string>    names(unk_->nComp());
    names[0] = "Density";
    names[1] = "Energy";

    amrex::WriteSingleLevelPlotfile(filename, *unk_, names, geometry_, 0.0, 0);
}

