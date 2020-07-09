#ifndef GRID_AXIS_H__
#define GRID_AXIS_H__

//Assumes {I,J,K}AXIS_C are macros from constants.h
#include "constants.h"

namespace orchestration {

//OPTION 1 - make Axis enum typed.
// PROS: Simple.
// CONS: Not typesafe and clogs namespace.
//Usage:
//   void foo(const Axis myAxis) { ... };
//
//   foo(iAx);
//   coord[iAx] = 0.0;
//     //OR
//   Axis ax = iAx;
//   foo(ax);
//   coord[ax] = 0.0;


enum Axis { iAx = IAXIS_C, jAx = JAXIS_C, kAx = KAXIS_C};


//OPTION 2 - make Axis an enum class.
// PROS: Simple, protects the namespace, typesafe.
// CONS: No implicit conversion to int.
//Usage:
//   void foo(const Axis myAxis) { ... };
//
//   foo(Axis::I);
//   coord[int(Axis::I)] = 0.0;
//     //OR
//   Axis ax = Axis::I;
//   foo(ax);
//   coord[int(ax)] = 0.0;

enum class Axis { I = IAXIS_C, J = JAXIS_C, K = KAXIS_C};


//OPTION 3 - make Axis a class with static members.
// PROS: Protects namespace, allows for implicit conversion to and from int.
// CONS: Kinda "reinventing the wheel". Potentially not performance optimized.
//Usage:
//   void foo(const Axis myAxis) { ... };
//
//   foo(Axis::I);
//   coord[Axis::I] = 0.0;
//     //OR
//   Axis ax = Axis::I;
//   foo(ax);
//   coord[ax] = 0.0;

class Axis{
    public:
        static constexpr int I = IAXIS_C;
        static constexpr int J = JAXIS_C;
        static constexpr int K = KAXIS_C;

        Axis(int vin) : value(vin) {
            if(vin!=I && vin!=J && vin!=K) throw std::logic_error("invalid value for axis.");
        }
        operator int() {
            return value;
        }
    private:
        int value;
};


} 
#endif
