// This code was generated with packet_generator.py.
#ifndef DATAPACKET_HYDRO_GPU_3_
#define DATAPACKET_HYDRO_GPU_3_
#include <Milhoja.h>
#include <Milhoja_DataPacket.h>
#include <Milhoja_RealVect.h>	
#include <Milhoja_IntVect.h>	
#include <Milhoja_FArray4D.h>	
#include <Milhoja_real.h>	
class DataPacket_Hydro_gpu_3 : public milhoja::DataPacket { 
public:
	std::unique_ptr<milhoja::DataPacket> clone(void) const override;
	DataPacket_Hydro_gpu_3(const milhoja::Real new_dt);
	~DataPacket_Hydro_gpu_3(void);
	DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&)                  = delete;
	DataPacket_Hydro_gpu_3(const DataPacket_Hydro_gpu_3&)            = delete;
	DataPacket_Hydro_gpu_3(DataPacket_Hydro_gpu_3&& packet)          = delete;
	DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&)       = delete;
	DataPacket_Hydro_gpu_3& operator=(const DataPacket_Hydro_gpu_3&) = delete;
	DataPacket_Hydro_gpu_3& operator=(DataPacket_Hydro_gpu_3&& rhs)  = delete;
	void pack(void) override;
	void unpack(void) override;
	int* nTiles_devptr(void) const { return static_cast<int*>(nTiles_start_d_); }
	milhoja::Real* dt_devptr(void) const { return static_cast<milhoja::Real*>(dt_start_d_); }
	milhoja::RealVect* deltas_devptr(void) const { return static_cast<milhoja::RealVect*>(deltas_start_d_); }
	milhoja::IntVect* lo_devptr(void) const { return static_cast<milhoja::IntVect*>(lo_start_d_); }
	milhoja::IntVect* hi_devptr(void) const { return static_cast<milhoja::IntVect*>(hi_start_d_); }
	milhoja::Real* CC1_devptr(void) const { return static_cast<milhoja::Real*>(CC1_start_d_); }
	milhoja::Real* CC2_devptr(void) const { return static_cast<milhoja::Real*>(CC2_start_d_); }
	milhoja::Real* FCX_devptr(void) const { return static_cast<milhoja::Real*>(FCX_start_d_); }
	milhoja::Real* FCY_devptr(void) const { return static_cast<milhoja::Real*>(FCY_start_d_); }
private:
	static constexpr std::size_t ALIGN_SIZE=16;
	static constexpr std::size_t pad(const std::size_t size) { return ((size + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE; }
	const milhoja::Real dt;
	std::size_t nTiles_BLOCK_SIZE_HELPER = 0;
	std::size_t dt_BLOCK_SIZE_HELPER = 0;
	std::size_t deltas_BLOCK_SIZE_HELPER = 0;
	std::size_t lo_BLOCK_SIZE_HELPER = 0;
	std::size_t hi_BLOCK_SIZE_HELPER = 0;
	std::size_t CC1_BLOCK_SIZE_HELPER = 0;
	std::size_t CC2_BLOCK_SIZE_HELPER = 0;
	std::size_t FCX_BLOCK_SIZE_HELPER = 0;
	std::size_t FCY_BLOCK_SIZE_HELPER = 0;
	int nTiles;
	void* nTiles_start_p_ = 0;
	void* nTiles_start_d_ = 0;
	void* dt_start_p_ = nullptr;
	void* dt_start_d_ = nullptr;
	void* deltas_start_p_ = nullptr;
	void* deltas_start_d_ = nullptr;
	void* lo_start_p_ = nullptr;
	void* lo_start_d_ = nullptr;
	void* hi_start_p_ = nullptr;
	void* hi_start_d_ = nullptr;
	void* CC1_start_p_ = nullptr;
	void* CC1_start_d_ = nullptr;
	void* CC2_start_d_ = nullptr;
	void* FCX_start_d_ = nullptr;
	void* FCY_start_d_ = nullptr;
};
#endif
