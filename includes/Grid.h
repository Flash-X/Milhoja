/**
 * \file    Grid.h
 *
 * \brief 
 *
 */

#ifndef GRID_H__
#define GRID_H__

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

using SET_IC_FCN = void (*)(void);

template<unsigned int NX,unsigned int NY,unsigned int NZ,unsigned int NGC>
class Grid {
public:
    ~Grid(void);

    static Grid* instance(void);

    void    initDomain(const amrex::Real xMin, const amrex::Real xMax,
                       const amrex::Real yMin, const amrex::Real yMax,
                       const amrex::Real zMin, const amrex::Real zMax,
                       const unsigned int nBlocksX,
                       const unsigned int nBlocksY,
                       const unsigned int nBlocksZ,
                       SET_IC_FCN initBlock);
    void    destroyDomain(void);

    amrex::MultiFab&   unk(void)       { return (*unk_); }
    amrex::Geometry&   geometry(void)  { return geometry_; }

private:
    Grid(void);
    Grid(const Grid&) = delete;
    Grid(const Grid&&) = delete;
    Grid& operator=(const Grid&) = delete;
    Grid& operator=(const Grid&&) = delete;

    static Grid*       instance_;

    amrex::Geometry    geometry_;
    amrex::MultiFab*   unk_;
};

#include "../src/Grid.cpp"

#endif

