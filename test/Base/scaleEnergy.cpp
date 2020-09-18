#include "scaleEnergy.h"

#include "Flash.h"

void StaticPhysicsRoutines::scaleEnergy(const orchestration::IntVect& lo,
                                        const orchestration::IntVect& hi,
                                        const orchestration::FArray1D& xCoords,
                                        const orchestration::FArray1D& yCoords,
                                        orchestration::FArray4D& f,
                                        const orchestration::Real scaleFactor) {
    // OFFLINE TOOLCHAIN: Add in directives for mapping to kernel
    orchestration::Real    x = 0.0;
    orchestration::Real    y = 0.0;
    for         (int k=lo.K(); k<=hi.K(); ++k) {
        for     (int j=lo.J(); j<=hi.J(); ++j) {
            y = yCoords(j);
            for (int i=lo.I(); i<=hi.I(); ++i) {
                x = xCoords(i);
                f(i, j, k, ENER_VAR_C) *= scaleFactor * x * y;
            }
        }
    }
}

