#ifdef DEBUG_RUNTIME
#include <cstdio>
#include <string>
#endif

#include "Milhoja.h"
#include "Milhoja_TileAmrex.h"

#include <AMReX_FArrayBox.H>

#include "Milhoja_Logger.h"
#include "Milhoja_Grid.h"

namespace milhoja {

/**
 * \brief Constructor for TileAmrex
 *
 * Should be called from inside a Tile Iterator, specifically:
 * TileIterAmrex::buildCurrentTile. Initializes private members.
 *
 * TODO:  Include a single metadata routine that gets gId, level,
 *        lo/hi, and loGC/hiGC in one call?  This could replace the
 *        lo(int*), etc. calls.
 *
 * \param itor An AMReX MFIter currently iterating.
 * \param unkRef A ref to the multifab being iterated over.
 * \param level Level of iterator.
 */
TileAmrex::TileAmrex(amrex::MFIter& itor,
                     amrex::MultiFab& unkRef,
                     const unsigned int level)
    : Tile{},
      level_{level},
      gridIdx_{ itor.index() },
      nCcVars_{0},
      dataPtr_{nullptr}
{
    amrex::FArrayBox& fab = unkRef[gridIdx_];
    int   nComp = fab.nComp();
    // TODO: Acceptable to have no variables?  Accept zero
    // here as a test for valid casting, but put a non-zero check in isNull?
    assert(nComp >= 0);
    nCcVars_ = static_cast<unsigned int>(nComp);

    amrex::Box   interior = itor.validbox();
    amrex::Box   GC       = itor.fabbox();
    const int*   loVect   = interior.loVect();
    const int*   hiVect   = interior.hiVect();
    const int*   loGCVect = GC.loVect();
    const int*   hiGCVect = GC.hiVect();

    dataPtr_ = static_cast<Real*>(fab.dataPtr());

    lo_[0]   = loVect[0];
    hi_[0]   = hiVect[0];
    loGC_[0] = loGCVect[0];
    hiGC_[0] = hiGCVect[0];
#if MILHOJA_NDIM >= 2
    lo_[1]   = loVect[1];
    hi_[1]   = hiVect[1];
    loGC_[1] = loGCVect[1];
    hiGC_[1] = hiGCVect[1];
#else
    lo_[1]   = 0;
    hi_[1]   = 0;
    loGC_[1] = 0;
    hiGC_[1] = 0;
#endif
#if MILHOJA_NDIM == 3
    lo_[2]   = loVect[2];
    hi_[2]   = hiVect[2];
    loGC_[2] = loGCVect[2];
    hiGC_[2] = hiGCVect[2];
#else
    lo_[2]   = 0;
    hi_[2]   = 0;
    loGC_[2] = 0;
    hiGC_[2] = 0;
#endif

// DEBUG TOOLS: Setup tile-specific data without having
//              to get too much data from AMReX.
//    loGC_[0] = gridIdx_ + -4;
//    loGC_[1] = gridIdx_ +  4;
//    loGC_[2] = gridIdx_ + 12;
//    lo_[0]   = gridIdx_ +  0;
//    lo_[1]   = gridIdx_ +  8;
//    lo_[2]   = gridIdx_ + 16;
//    hi_[0]   = gridIdx_ +  7;
//    hi_[1]   = gridIdx_ + 15;
//    hi_[2]   = gridIdx_ + 23;
//    hiGC_[0] = gridIdx_ + 11;
//    hiGC_[1] = gridIdx_ + 19;
//    hiGC_[2] = gridIdx_ + 27;

#ifdef DEBUG_RUNTIME
    // This was useful for developing the Fortran/C interoperability layer
    printf("TileAmrex::TileAmrex] Tile %d (%p) / level=%d / lo=(%d,%d,%d) / hi=(%d,%d,%d) / loGC=(%d,%d,%d) / hiGC=(%d,%d,%d) / dataPtr %p\n",
           gridIdx_, this, level_, 
             lo_[0],   lo_[1],   lo_[2],
             hi_[0],   hi_[1],   hi_[2],
           loGC_[0], loGC_[1], loGC_[2],
           hiGC_[0], hiGC_[1], hiGC_[2],
           dataPtr_);

    std::string   msg = "[TileAmrex] Created Tile object "
                  + std::to_string(gridIdx_)
                  + " from MFIter";
    Logger::instance().log(msg);
#endif
}

/**
 * \brief Destructor for TileAmrex
 *
 * Deletes/nullifies private members.
 */
TileAmrex::~TileAmrex(void) {
    dataPtr_ = nullptr;
#ifdef DEBUG_RUNTIME
    std::string msg = "[TileAmrex] Destroying Tile object "
                      + std::to_string(gridIdx_);
    Logger::instance().log(msg);
#endif
}

/**
 * \todo Is there a valid use case for creating null Tile's?
 * If so, best to put that motivation in the documentation.  After
 * changes, is this the correct definition of null?  Include asserts
 * that confirm that correct ordering between lo, hi, loGC, and hiGC?
 *
 * \brief Checks whether a Tile is null.
 */
bool   TileAmrex::isNull(void) const {
    return (   (gridIdx_ < 0) //TODO this is never true?
            && (level_ == 0)
            && (!dataPtr_)
            && (nCcVars_ <= 0) );
}

/**
 * These are Fortran friendly since we skip the IntVect.  It's also
 * Flash-X friendly since you get MDIM sized points.  Not clear if
 * this is really necessary.
 * \TODO Would it be better to simply get a pointer to the
 * start of the underlying amrex::IntVect data buffer and
 * return this?
 */
void   TileAmrex::lo(int* i, int* j, int* k) const {
    *i = lo_[0];
    *j = lo_[1];
    *k = lo_[2];
#ifdef DEBUG_RUNTIME
    // This was useful for developing the Fortran/C interoperability layer
    printf("TileAmrex::lo      ] Tile %d (%p) / level=%d / lo=(%d,%d,%d) / hi=(%d,%d,%d) / loGC=(%d,%d,%d) / hiGC=(%d,%d,%d)\n",
           gridIdx_, this, level_, 
                 *i,       *j,       *k,
             hi_[0],   hi_[1],   hi_[2],
           loGC_[0], loGC_[1], loGC_[2],
           hiGC_[0], hiGC_[1], hiGC_[2]);
#endif
}

void   TileAmrex::hi(int* i, int* j, int* k) const {
    *i = hi_[0];
    *j = hi_[1];
    *k = hi_[2];
#ifdef DEBUG_RUNTIME
    // This was useful for developing the Fortran/C interoperability layer
    printf("TileAmrex::hi      ] Tile %d (%p) / level=%d / lo=(%d,%d,%d) / hi=(%d,%d,%d) / loGC=(%d,%d,%d) / hiGC=(%d,%d,%d)\n",
           gridIdx_, this, level_, 
             lo_[0],   lo_[1],   lo_[2],
                 *i,       *j,       *k,
           loGC_[0], loGC_[1], loGC_[2],
           hiGC_[0], hiGC_[1], hiGC_[2]);
#endif
}

void   TileAmrex::loGC(int* i, int* j, int* k) const {
    *i = loGC_[0];
    *j = loGC_[1];
    *k = loGC_[2];
#ifdef DEBUG_RUNTIME
    // This was useful for developing the Fortran/C interoperability layer
    printf("TileAmrex::loGC    ] Tile %d (%p) / level=%d / lo=(%d,%d,%d) / hi=(%d,%d,%d) / loGC=(%d,%d,%d) / hiGC=(%d,%d,%d)\n",
           gridIdx_, this, level_, 
             lo_[0],   lo_[1],   lo_[2],
             hi_[0],   hi_[1],   hi_[2],
                 *i,       *j,       *k,
           hiGC_[0], hiGC_[1], hiGC_[2]);
#endif
}

void   TileAmrex::hiGC(int* i, int* j, int* k) const {
    *i = hiGC_[0];
    *j = hiGC_[1];
    *k = hiGC_[2];
#ifdef DEBUG_RUNTIME
    // This was useful for developing the Fortran/C interoperability layer
    printf("TileAmrex::hiGC    ] Tile %d (%p) / level=%d / lo=(%d,%d,%d) / hi=(%d,%d,%d) / loGC=(%d,%d,%d) / hiGC=(%d,%d,%d)\n",
           gridIdx_, this, level_, 
             lo_[0],   lo_[1],   lo_[2],
             hi_[0],   hi_[1],   hi_[2],
           loGC_[0], loGC_[1], loGC_[2],
                 *i,       *j,       *k);
#endif
}

/**
 * \brief Gets index of lo cell in the Tile
 *
 * \return IntVect with index of lower left cell.
 */
IntVect  TileAmrex::lo(void) const {
    return IntVect(LIST_NDIM(lo_[0], lo_[1], lo_[2]));
}

/**
 * \brief Gets index of hi cell in the Tile
 *
 * \return IntVect with index of upper right cell.
 */
IntVect  TileAmrex::hi(void) const {
    return IntVect(LIST_NDIM(hi_[0], hi_[1], hi_[2]));
}

/**
 * \brief Gets index of lo guard cell in the Tile
 *
 * \return IntVect with index of lower left cell, including
 *         guard cells.
 */
IntVect  TileAmrex::loGC(void) const {
    return IntVect(LIST_NDIM(loGC_[0], loGC_[1], loGC_[2]));
}

/**
 * \brief Gets index of hi guard cell in the Tile
 *
 * \return IntVect with index of upper right cell, including
 *         guard cells.
 */
IntVect  TileAmrex::hiGC(void) const {
    return IntVect(LIST_NDIM(hiGC_[0], hiGC_[1], hiGC_[2]));
}

/**
 * \brief Returns pointer to underlying data structure.
 *
 * \TODO This routine should return the lo/hi and shape of the data associated
 *       with the pointer.  AMReX dictates what we point to and this is
 *       analogous to wrapping the data with FArray4D.
 *
 * \return Pointer to start of tile's data in host memory.
 */
Real*   TileAmrex::dataPtr(void) {
#ifdef DEBUG_RUNTIME
    // This was useful for developing the Fortran/C interoperability layer
    printf("TileAmrex::dataPtr] Tile %d (%p) / level=%d / lo=(%d,%d,%d) / hi=(%d,%d,%d) / loGC=(%d,%d,%d) / hiGC=(%d,%d,%d) / dataPtr %p\n",
           gridIdx_, this, level_, 
             lo_[0],   lo_[1],   lo_[2],
             hi_[0],   hi_[1],   hi_[2],
           loGC_[0], loGC_[1], loGC_[2],
           hiGC_[0], hiGC_[1], hiGC_[2],
           dataPtr_);
#endif
    return dataPtr_;
}

/**
 * \brief Returns FArray4D to access underlying data.
 *
 * \return A FArray4D object which wraps the pointer to underlying
 *         data and provides Fortran-style access.
 */
FArray4D TileAmrex::data(void) {
    return FArray4D{dataPtr(), loGC(), hiGC(), nCcVars_};
}

}

