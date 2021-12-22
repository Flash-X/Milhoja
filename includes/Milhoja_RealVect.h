#ifndef MILHOJA_REALVECT_H__
#define MILHOJA_REALVECT_H__

#include <iosfwd>
#include <stdexcept>

#include "milhoja.h"
#include "Milhoja_macros.h"
#include "Milhoja_axis.h"
#include "Milhoja_real.h"

#ifdef GRID_AMREX
#include <AMReX_RealVect.H>
#endif

namespace milhoja {

class IntVect;

/**
  * \brief Container for NDIM tuples of Reals.
  *
  * TODO detailed description.
  */
class RealVect
{
  public:

    /** \brief Default constructor
      *
      * Returns a vector with undefined components.
      * TODO: return a default value aka 0?
      */
    explicit RealVect () {}

    //! Constructor from NDIM Reals.
    constexpr explicit RealVect (LIST_NDIM(const Real x, const Real y, const Real z))
        : LIST_NDIM(i_{x}, j_{y}, k_{z}) {}

    //! Constructor from Real*.
    explicit RealVect (const Real* x)
        : LIST_NDIM(i_{x[0]}, j_{x[1]}, k_{x[2]}) {}

#if NDIM<3
    //! Constructor from 3 Reals.
    explicit RealVect (const Real x, const Real y, const Real z)
        : LIST_NDIM(i_{x}, j_{y}, k_{z}) {
        throw std::logic_error("Using deprecated RealVect constructor. Please wrap arguments in LIST_NDIM macro.\n");
    }
#endif

#ifdef GRID_AMREX
    //! Constructor from amrex::RealVect
    explicit RealVect (const amrex::RealVect& ain)
        : LIST_NDIM(i_{ain[0]},j_{ain[1]},k_{ain[2]}) {}

    //! Operator to explicitly cast an RealVect to an AMReX RealVect
    explicit operator amrex::RealVect () const {
        return amrex::RealVect(LIST_NDIM(i_,j_,k_));
    }
#endif

    // Allow move semantics but no copies.
    RealVect(RealVect&&) = default;
    RealVect& operator=(RealVect&&) = default;
    RealVect(RealVect&) = delete;
    RealVect(const RealVect&) = delete;
    RealVect& operator=(RealVect&) = delete;
    RealVect& operator=(const RealVect&) = delete;

    // Functions to convert to IntVects.
    IntVect round() const;
    IntVect floor() const;
    IntVect ceil() const;

    //! Return first element of vector
    Real I() const {
        return i_;
    }

    //! Return second element of vector, or 0 if NDIM<2
    Real J() const {
#if (NDIM>=2)
        return j_;
#else
        return 0.0_wp;
#endif
    }

    //! Return third element of vector, or 0 if NDIM<3
    Real K() const {
#if (NDIM==3)
        return k_;
#else
        return 0.0_wp;
#endif
    }

    /** \brief Get and set values of the internal array.
      * Perform bounds check unless GRID_ERRCHECK_OFF is set.
      */
    Real& operator[] (const int i) {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in RealVect.");
        }
#endif
        switch(i) {
            case Axis::I:
                return i_;
#if NDIM>=2
            case Axis::J:
                return j_;
#endif
#if NDIM==3
            case Axis::K:
                return k_;
#endif
        }
        return i_;
    }
    //! Get values of the internal array as consts.
    const Real& operator[] (const int i) const {
#ifndef GRID_ERRCHECK_OFF
        if(i>=NDIM || i<0) {
            throw std::logic_error("Index out-of-bounds in RealVect.");
        }
#endif
        switch(i) {
            case Axis::I:
                return i_;
#if NDIM>=2
            case Axis::J:
                return j_;
#endif
#if NDIM==3
            case Axis::K:
                return k_;
#endif
        }
        return i_;
    }

    //TODO: Potential operators
    // ==, != scalar
    // >, <, etc
    // unary +, -
    // +=, -=, *=, /=
    // + scalar, - scalar
    // min, max, etc

    //! Add vectors component-wise.
    RealVect operator+ (const RealVect& b) const {
      return RealVect(LIST_NDIM(i_+b.I(), j_+b.J(), k_+b.K()));
    }

    //! Subtract vectors component-wise.
    RealVect operator- (const RealVect& b) const {
      return RealVect(LIST_NDIM(i_-b.I(), j_-b.J(), k_-b.K()));
    }

    //! Multiply two vectors component-wise.
    RealVect operator* (const RealVect& b) const {
      return RealVect(LIST_NDIM(i_*b.I(), j_*b.J(), k_*b.K()));
    }

    //! Multiply a vector by a scalar (V * c).
    RealVect operator* (const Real c) const {
      return RealVect(LIST_NDIM(i_*c, j_*c, k_*c));
    }

    //! Divide two vectors component-wise.
    RealVect operator/ (const RealVect& b) const {
      return RealVect(LIST_NDIM(i_/b.I(), j_/b.J(), k_/b.K()));
    }

    //! Divide a vector by a scalar.
    RealVect operator/ (const Real c) const {
      return (*this)*(1.0_wp/c);
    }

    //! Return product of elements.
    Real product() const {
      return CONCAT_NDIM(i_, * j_, * k_);
    }

    //! Return pointer to underlying array
    //const Real* dataPtr() const {
    //  return vect_;
    //}

    friend std::ostream& operator<< (std::ostream& os, const RealVect& vout);

  private:

    Real LIST_NDIM(i_,j_,k_);   //!< Contains data.
};

//! Scalar multiply a vector (c * V).
RealVect operator* (const Real c, const RealVect& a);


}
#endif
