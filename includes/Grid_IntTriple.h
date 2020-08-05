#ifndef GRID_INTTRIPLE_H__
#define GRID_INTTRIPLE_H__

#include "constants.h"
#include "Grid_Macros.h"
#include <iosfwd>
#include <stdexcept>

#ifdef GRID_AMREX
#include <AMReX_Dim3.H>
#include <AMReX_IntVect.H>
#endif

namespace orchestration {

/**
  * \brief Container for MDIM-tuples of integers.
  *
  * A very "weak" class - it has no math operators defined and cannot
  * be converted into any other data type. Should be used only for
  * triple nested loops and for defining bounds of FArray4Ds.
  *
  */
class IntTriple
{
  public:

    /** \brief Default constructor
      *
      * Returns a vector with undefined components.
      * TODO: return a default value aka 0?
      */
    explicit IntTriple () {}

    //! Constructor from MDIM ints.
    constexpr explicit IntTriple (const int x, const int y, const int z)
        : vect_{x,y,z} {}

#ifdef GRID_AMREX
    //! Constructor from amrex::Dim3
    explicit IntTriple (const amrex::Dim3& ain) : vect_{ain.x,ain.y,ain.z} {}

    //! Operator to explicitly cast an IntTriple to an AMReX Dim3
    explicit operator amrex::Dim3 () const {
        return amrex::IntVect(LIST_NDIM(vect_[0],vect_[1],vect_[2])).dim3() ;
    }
#endif

    // Allow move semantics but no copies.
    IntTriple(IntTriple&&) = default;
    IntTriple& operator=(IntTriple&&) = default;
    IntTriple(IntTriple&) = delete;
    IntTriple(const IntTriple&) = delete;
    IntTriple& operator=(IntTriple&) = delete;
    IntTriple& operator=(const IntTriple&) = delete;

    //! Get and set values of the internal array.
    int& operator[] (const int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=MDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in IntTriple.");
        }
#endif
        return vect_[i];
    }
    //! Get values of the internal array as consts.
    const int& operator[] (const int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=MDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in IntTriple.");
        }
#endif
        return vect_[i];
    }

    // TODO pretty printing
    //friend std::ostream& operator<< (std::ostream& os, const IntTriple& vout);

  private:
    int vect_[MDIM];   //!< Contains data.
};

}
#endif
