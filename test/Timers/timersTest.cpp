#include "gtest/gtest.h"
#include "Timers.h"

#include <iostream>

using namespace orchestration;

namespace {

class TimersTest : public testing::Test {
protected:
    TimersTest(void) {
    }

    ~TimersTest(void) {
    }
};

TEST_F(TimersTest,Sample){
    Timers::start("evolution");
        Timers::start("Hydro");
            Timers::start("Head");
                Timers::start("guardcell Barrier");
                Timers::stop("guardcell Barrier");
                Timers::start("guardcell internal");
                    Timers::start("amr_guardcell");
                    Timers::stop("amr_guardcell");
                    Timers::start("eos gc");
                    Timers::stop("eos gc");
                Timers::stop("guardcell internal");
            Timers::stop("Head");
            Timers::start("compute fluxes");
            Timers::stop("compute fluxes");
                Timers::start("hy_computeFluxes");
                Timers::stop("hy_computeFluxes");
            Timers::start("conserveFluxes");
            Timers::stop("conserveFluxes");
            Timers::start("update solution");
            Timers::stop("update solution");
        Timers::stop("Hydro");
        Timers::start("sourceTerms");
        Timers::stop("sourceTerms");


        Timers::start("Hydro");
            Timers::start("Head");
                Timers::start("guardcell Barrier");
                Timers::stop("guardcell Barrier");
                Timers::start("guardcell internal");
                    Timers::start("amr_guardcell");
                    Timers::stop("amr_guardcell");
                    Timers::start("eos gc");
                    Timers::stop("eos gc");
                Timers::stop("guardcell internal");
            Timers::stop("Head");
            Timers::start("compute fluxes");
            Timers::stop("compute fluxes");
                Timers::start("hy_computeFluxes");
                    Timers::start("RiemannState");
                    Timers::stop("RiemannState");
                    Timers::start("getFaceFlux");
                    Timers::stop("getFaceFlux");
                    Timers::start("unsplitUpdate");
                    Timers::stop("unsplitUpdate");
                Timers::stop("hy_computeFluxes");
            Timers::start("conserveFluxes");
            Timers::stop("conserveFluxes");
            Timers::start("update solution");
                Timers::start("update solution body");
                    Timers::start("unsplitUpdate");
                    Timers::stop("unsplitUpdate");
                Timers::stop("update solution body");
            Timers::stop("update solution");
        Timers::stop("Hydro");
        Timers::start("sourceTerms");
        Timers::stop("sourceTerms");
    Timers::stop("evolution");
}

}
