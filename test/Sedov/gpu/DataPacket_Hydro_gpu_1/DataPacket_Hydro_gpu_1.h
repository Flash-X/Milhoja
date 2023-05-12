// This code was generated with packet_generator.py.
#ifndef DATAPACKET_HYDRO_GPU_1_
#define DATAPACKET_HYDRO_GPU_1_
#include <Milhoja.h>
#include <Milhoja_DataPacket.h>
#include <Milhoja_RealVect.h>	
#include <Milhoja_real.h>	
#include <Milhoja_FArray4D.h>	
#include <Milhoja_IntVect.h>	
class DataPacket_Hydro_gpu_1 : public milhoja::DataPacket { 
public:
	std::unique_ptr<milhoja::DataPacket> clone(void) const override;
	DataPacket_Hydro_gpu_1(const milhoja::Real new_dt);
	~DataPacket_Hydro_gpu_1(void);
	DataPacket_Hydro_gpu_1(DataPacket_Hydro_gpu_1&)                  = delete;
	DataPacket_Hydro_gpu_1(const DataPacket_Hydro_gpu_1&)            = delete;
	DataPacket_Hydro_gpu_1(DataPacket_Hydro_gpu_1&& packet)          = delete;
	DataPacket_Hydro_gpu_1& operator=(DataPacket_Hydro_gpu_1&)       = delete;
	DataPacket_Hydro_gpu_1& operator=(const DataPacket_Hydro_gpu_1&) = delete;
	DataPacket_Hydro_gpu_1& operator=(DataPacket_Hydro_gpu_1&& rhs)  = delete;
	void pack(void) override;
	void unpack(void) override;
	std::size_t* nTiles_getter(void) const { return static_cast<std::size_t*>(nTiles_start_d_); }
	milhoja::Real* dt_getter(void) const { return static_cast<milhoja::Real*>(dt_start_d_); }
	milhoja::RealVect* deltas_getter(void) const { return static_cast<milhoja::RealVect*>(deltas_start_d_); }
	milhoja::IntVect* lo_getter(void) const { return static_cast<milhoja::IntVect*>(lo_start_d_); }
	milhoja::IntVect* hi_getter(void) const { return static_cast<milhoja::IntVect*>(hi_start_d_); }
	milhoja::Real* CC1_getter(void) const { return static_cast<milhoja::Real*>(CC1_start_d_); }
	milhoja::Real* CC2_getter(void) const { return static_cast<milhoja::Real*>(CC2_start_d_); }
	milhoja::Real* FCX_getter(void) const { return static_cast<milhoja::Real*>(FCX_start_d_); }
	milhoja::Real* FCY_getter(void) const { return static_cast<milhoja::Real*>(FCY_start_d_); }
	int extraAsynchronousQueue(const unsigned int id) override;
	void releaseExtraQueue(const unsigned int id) override;
private:
	static const unsigned int EXTRA_STREAMS = 1;
	milhoja::Stream stream2_;
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
	void* CC2_start_p_ = nullptr;
	void* CC2_start_d_ = nullptr;
	void* FCX_start_d_ = nullptr;
	void* FCY_start_d_ = nullptr;
};
#endif
