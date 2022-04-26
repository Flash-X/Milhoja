#ifndef MILHOJA_EXT_BC_FILL_AMREX_H__
#define MILHOJA_EXT_BC_FILL_AMREX_H__

#include <iostream>
#include <stdexcept>

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BCRec.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>

#include "Milhoja_boundaryConditions.h"

namespace milhoja {

/**
 * \todo Write documentation.
 */
class ExtBcFillAMReX {
public:
    /**
     * \param level               The refinement level of the BC data needed
     * \param externalBcRoutine   The external routine that should handle non-periodic
     *                            BC.  This pointer can be nullptr if all BC are
     *                            periodic.  However, it is a logical error to call the
     *                            function wrapped by the object in this case.
     */
    ExtBcFillAMReX(const int level, BC_ROUTINE externalBcRoutine)
        : level_{level},
          externalBcRoutine_{externalBcRoutine}
    {}
    ~ExtBcFillAMReX(void) {}

    ExtBcFillAMReX(ExtBcFillAMReX&)                  = delete;
    ExtBcFillAMReX(const ExtBcFillAMReX&)            = default;
    ExtBcFillAMReX(ExtBcFillAMReX&&)                 = delete;
    ExtBcFillAMReX& operator=(ExtBcFillAMReX&)       = delete;
    ExtBcFillAMReX& operator=(const ExtBcFillAMReX&) = delete;
    ExtBcFillAMReX& operator=(ExtBcFillAMReX&&)      = delete;

    /**
     * This is the interface defined for the AMReX CpuBndryFuncFab wrapper
     * class.
     *
     * \todo How would this work if the data is in the GPU?
     * \todo Is this the right level of function wrapper or do we want to go one
     *       level higher?
     * \todo Figure out what all the arguments are for and how they should be
     *       used.  Document them here carefully and determine the interface of
     *       BC_ROUTINE.
     * \todo Can we make use of AMReX low-level functions to help identify
     *       the regions that need BC?  Pass the domain?
     */
    void operator() (amrex::Box const& box,
                     amrex::FArrayBox& dest,
                     const int dcomp,
                     const int numcomp,
                     amrex::Geometry const& geom,
                     const amrex::Real time,
                     const amrex::Vector<amrex::BCRec>& bcr,
                     const int bcomp,
                     const int orig_comp) {
        if (!externalBcRoutine_) {
            throw std::logic_error("[ExtBcFillAMReX::op()] Null BC function pointer");
        }

        const int*   lo = box.loVect();
        const int*   hi = box.hiVect();

        externalBcRoutine_(lo, hi, level_, dcomp, numcomp);
    }

private:
    const int          level_;
    const BC_ROUTINE   externalBcRoutine_;
};

}

#endif

