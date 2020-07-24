/**
 * \file    GridAmrex.h
 *
 * \brief
 *
 */

#ifndef GRIDAMREX_H__
#define GRIDAMREX_H__

#include "Grid.h"

#include "Grid_RealVect.h"

namespace orchestration {

class GridAmrex : public Grid {
public:
    RealVect       getProbLo() const override;
    RealVect       getProbHi() const override;

private:
    friend Grid& Grid::instance();
    GridAmrex(void) {}

    GridAmrex(const GridAmrex&) = delete;
    GridAmrex(GridAmrex&&) = delete;
    GridAmrex& operator=(const GridAmrex&) = delete;
    GridAmrex& operator=(GridAmrex&&) = delete;
};

}

#endif

