// This code was generated with packet_generator.py.
#include "HydroPacket3Test.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_IntVect.h>
#include "Sedov.h"
#include <Driver.h>
HydroPacket3Test::HydroPacket3Test(void) : milhoja::DataPacket(){}, 
	dt{0},
	n{0},
	CC1_BLOCK_SIZE{0},
	CC2_BLOCK_SIZE{0},
	FCX_BLOCK_SIZE{0},
	FCY_BLOCK_SIZE{0},
	FCZ_BLOCK_SIZE{0},
	deltas{0},
	deltas_BLOCK_SIZE{0},
	lo{0},
	lo_BLOCK_SIZE{0},
	hi{0},
	hi_BLOCK_SIZE{0}
{
	using namespace milhoja;
	unsigned int nxb = 1;
	unsigned int nyb = 1;
	unsigned int nzb = 1;
	Grid::instance().getBlockSize(&nxb, &nyb, &nzb);
	FCX_BLOCK_SIZE = sizeof(Real) * (nxb+1) * nyb * nzb * NFLUXES;
	CC2_BLOCK_SIZE = (nxb + 2 * NGUARD * MILHOJA_K1D) * (nxy + 2 * NGUARD * MILHOJA_K2D) * (nxz + 2 * NGUARD * MILHOJA_K3D);
	FCZ_BLOCK_SIZE = sizeof(Real) * nxb * nyb * (nzb+1) * NFLUXES;
	FCY_BLOCK_SIZE = sizeof(Real) * nxb * (nyb+1) * nzb * NFLUXES;
	CC1_BLOCK_SIZE = (nxb + 2 * NGUARD * MILHOJA_K1D) * (nxy + 2 * NGUARD * MILHOJA_K2D) * (nxz + 2 * NGUARD * MILHOJA_K3D);
}
~HydroPacket3Test::HydroPacket3Test(void) {
}
void HydroPacket3Test::unpack(void) {
	using namespace milhoja;
}
void HydroPacket3Test::pack(void) {
	using namespace milhoja;
}
