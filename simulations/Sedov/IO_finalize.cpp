#include "IO.h"

/**
 * Finalize the IO unit.
 */
void   IO::finalize(void) {
    if (io::blockIntegralQuantities_mass) {
        delete [] io::blockIntegralQuantities_mass;
        io::blockIntegralQuantities_mass = nullptr;
    }

    if (io::blockIntegralQuantities_xmom) {
        delete [] io::blockIntegralQuantities_xmom;
        io::blockIntegralQuantities_xmom = nullptr;
    }

    if (io::blockIntegralQuantities_ymom) {
        delete [] io::blockIntegralQuantities_ymom;
        io::blockIntegralQuantities_ymom = nullptr;
    }

    if (io::blockIntegralQuantities_zmom) {
        delete [] io::blockIntegralQuantities_zmom;
        io::blockIntegralQuantities_zmom = nullptr;
    }

    if (io::blockIntegralQuantities_ener) {
        delete [] io::blockIntegralQuantities_ener;
        io::blockIntegralQuantities_ener = nullptr;
    }

    if (io::blockIntegralQuantities_ke) {
        delete [] io::blockIntegralQuantities_ke;
        io::blockIntegralQuantities_ke = nullptr;
    }

    if (io::blockIntegralQuantities_eint) {
        delete [] io::blockIntegralQuantities_eint;
        io::blockIntegralQuantities_eint = nullptr;
    }

    if (io::blockIntegralQuantities_magp) {
        delete [] io::blockIntegralQuantities_magp;
        io::blockIntegralQuantities_magp = nullptr;
    }

    if (localIntegralQuantities) {
        delete [] localIntegralQuantities;
        localIntegralQuantities = nullptr;
    }

    if (globalIntegralQuantities) {
        delete [] globalIntegralQuantities;
        globalIntegralQuantities = nullptr;
    }
}

