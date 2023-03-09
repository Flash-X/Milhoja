// This code was generated with packet_generator.py.
#ifndef HYDROPACKET3TEST
#define HYDROPACKET3TEST
#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_DataPacket.h>
class HydroPacket3Test : public milhoja::DataPacket { 
public:
	std::unique_ptr<milhoja::DataPacket> clone(void) const override;
	HydroPacket3Test(void);
	~HydroPacket3Test(void);
	HydroPacket3Test(HydroPacket3Test&)                  = delete;
	HydroPacket3Test(const HydroPacket3Test&)            = delete;
	HydroPacket3Test(HydroPacket3Test&& packet)          = delete;
	HydroPacket3Test& operator=(HydroPacket3Test&)       = delete;
	HydroPacket3Test& operator=(const HydroPacket3Test&) = delete;
	HydroPacket3Test& operator=(HydroPacket3Test&& rhs)  = delete;
	void pack(void) override;
	void unpack(void) override;
#if MILHOJA_NDIM == 3 && defined(MILHOJA_OPENACC_OFFLOADING)
	int extraAsynchronousQueue(const unsigned int id) override;
	void releaseExtraQueue(const unsigned int id) override;
#endif
private:
	milhoja::Real* dt;
	milhoja::Real* n;
	std::size_t N_ELEMENTS_PER_CC1_PER_VARIABLE;
	std::size_t N_ELEMENTS_PER_CC1;
	std::size_t CC1_BLOCK_SIZE_BYTES;
	std::size_t N_ELEMENTS_PER_CC2_PER_VARIABLE;
	std::size_t N_ELEMENTS_PER_CC2;
	std::size_t CC2_BLOCK_SIZE_BYTES;
	std::size_t N_ELEMENTS_PER_FCX_PER_VARIABLE;
	std::size_t N_ELEMENTS_PER_FCX;
	std::size_t FCX_BLOCK_SIZE_BYTES;
	std::size_t N_ELEMENTS_PER_FCY_PER_VARIABLE;
	std::size_t N_ELEMENTS_PER_FCY;
	std::size_t FCY_BLOCK_SIZE_BYTES;
	std::size_t N_ELEMENTS_PER_FCZ_PER_VARIABLE;
	std::size_t N_ELEMENTS_PER_FCZ;
	std::size_t FCZ_BLOCK_SIZE_BYTES;
	std::size_t RealVect_SIZE_BYTES;
	std::size_t IntVect_SIZE_BYTES;
	std::size_t FArray4D_SIZE_BYTES;
	std::size_t Real_SIZE_BYTES;
};
#endif
