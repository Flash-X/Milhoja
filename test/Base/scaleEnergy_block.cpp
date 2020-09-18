#include "scaleEnergy_block.h"

#include "Flash.h"

void ThreadRoutines::scaleEnergy_block(const orchestration::IntVect& lo,
                                       const orchestration::IntVect& hi,
                                       const orchestration::FArray1D& xCoords,
                                       const orchestration::FArray1D& yCoords,
                                       orchestration::FArray4D& f) {
    orchestration::Real    x = 0.0;
    orchestration::Real    y = 0.0;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            y = yCoords(j);
            for (int i=lo.I(); i<=hi.I(); ++i) {
                x = xCoords(i);
                f(i, j, k, ENER_VAR_C) *= 5.0 * x * y;
            }
        }
    }
}

