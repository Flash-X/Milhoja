// This code was generated with packet_generator.py.
#include "/autofs/nccs-svm1_home1/wkwiecinski/OrchestrationRuntime/test/Sedov/gpu/DataPacket_Hydro_gpu_3/DataPacket_Hydro_gpu_3.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <Milhoja_Grid.h>
#include <Milhoja_RuntimeBackend.h>
#include "Sedov.h"
#include "Driver.h"
DataPacket_Hydro_gpu_3::DataPacket_Hydro_gpu_3(const milhoja::Real new_dt) : milhoja::DataPacket{}, 
	dt_BLOCK_SIZE_HELPER{0},
	deltas_BLOCK_SIZE_HELPER{0},
	lo_BLOCK_SIZE_HELPER{0},
	hi_BLOCK_SIZE_HELPER{0},
	CC1_BLOCK_SIZE_HELPER{0},
	CC2_BLOCK_SIZE_HELPER{0},
	FCX_BLOCK_SIZE_HELPER{0},
	FCY_BLOCK_SIZE_HELPER{0},
	dt{new_dt}
{
	using namespace milhoja;
	unsigned int nxb = 1;
	unsigned int nyb = 1;
	unsigned int nzb = 1;
	Grid::instance().getBlockSize(&nxb, &nyb, &nzb);
	CC2_BLOCK_SIZE_HELPER = (nxb + 2 * NGUARD * MILHOJA_K1D) * (nyb + 2 * NGUARD * MILHOJA_K2D) * (nzb + 2 * NGUARD * MILHOJA_K3D) * ((0+0+1)) * sizeof(Real);
	FCX_BLOCK_SIZE_HELPER = ((nxb+1) + 2 * 0) * ((nyb) + 2 * 0) * ((nzb) + 2 * 0) * (4+0+1) * sizeof(Real);
	FCY_BLOCK_SIZE_HELPER = ((nxb) + 2 * 0) * ((nyb+1) + 2 * 0) * ((nzb) + 2 * 0) * (4+0+1) * sizeof(Real);
	dt_BLOCK_SIZE_HELPER = pad(sizeof(Real));
	deltas_BLOCK_SIZE_HELPER = sizeof(RealVect);
	lo_BLOCK_SIZE_HELPER = sizeof(IntVect);
	hi_BLOCK_SIZE_HELPER = sizeof(IntVect);
	CC1_BLOCK_SIZE_HELPER = (nxb + 2 * NGUARD * MILHOJA_K1D) * (nyb + 2 * NGUARD * MILHOJA_K2D) * (nzb + 2 * NGUARD * MILHOJA_K3D) * ((GAMC_VAR+UNK_VARS_BEGIN+1)) * sizeof(Real);
}

DataPacket_Hydro_gpu_3::~DataPacket_Hydro_gpu_3(void) {
	nullify();
}

std::unique_ptr<milhoja::DataPacket> DataPacket_Hydro_gpu_3::clone(void) const {
	return std::unique_ptr<milhoja::DataPacket>{ new DataPacket_Hydro_gpu_3{dt} };
}

void DataPacket_Hydro_gpu_3::pack(void) {
	using namespace milhoja;
	std::string errMsg = isNull();
	if (errMsg != "")
		throw std::logic_error("[DataPacket_Hydro_gpu_3::pack] " + errMsg);
	else if (tiles_.size() == 0)
		throw std::logic_error("[DataPacket_Hydro_gpu_3::pack] No tiles added.");
	
	Grid& grid = Grid::instance();
	nTiles = tiles_.size();
	nTiles_BLOCK_SIZE_HELPER = sizeof(int);

	/// SIZE DETERMINATION
	// Scratch section
	std::size_t nScratchPerTileBytes = CC2_BLOCK_SIZE_HELPER + FCX_BLOCK_SIZE_HELPER + FCY_BLOCK_SIZE_HELPER;
	unsigned int nScratchArrays = 3;
	std::size_t nScratchPerTileBytesPadded = pad(nTiles * nScratchPerTileBytes);
	if (nScratchPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error("[DataPacket_Hydro_gpu_3] Scratch padding failure");

	// non tile specific data
	std::size_t nCopyInBytes = nTiles_BLOCK_SIZE_HELPER + dt_BLOCK_SIZE_HELPER  + nTiles * sizeof(PacketContents);
	std::size_t nCopyInBytesPadded = pad(nCopyInBytes);
	if (nCopyInBytesPadded % ALIGN_SIZE != 0) throw std::logic_error("[DataPacket_Hydro_gpu_3] CopyIn padding failure");

	std::size_t nBlockMetadataPerTileBytes = nTiles * ( 0 + deltas_BLOCK_SIZE_HELPER + lo_BLOCK_SIZE_HELPER + hi_BLOCK_SIZE_HELPER );
	std::size_t nBlockMetadataPerTileBytesPadded = pad(nBlockMetadataPerTileBytes);
	if (nBlockMetadataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error("[DataPacket_Hydro_gpu_3] Metadata padding failure");

	std::size_t nCopyInDataPerTileBytes = (0) * nTiles;
	std::size_t nCopyInDataPerTileBytesPadded = pad(nCopyInDataPerTileBytes);
	if (nCopyInDataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error("[DataPacket_Hydro_gpu_3] CopyInPerTile padding failure");

	std::size_t nCopyInOutDataPerTileBytes = (CC1_BLOCK_SIZE_HELPER) * nTiles;
	std::size_t nCopyInOutDataPerTileBytesPadded = pad(nCopyInOutDataPerTileBytes);
	if (nCopyInOutDataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error("[DataPacket_Hydro_gpu_3] CopyInOutPerTile padding failure");

	std::size_t nCopyOutDataPerTileBytes = (0) * nTiles;
	std::size_t nCopyOutDataPerTileBytesPadded = pad(nCopyOutDataPerTileBytes);
	if (nCopyOutDataPerTileBytesPadded % ALIGN_SIZE != 0) throw std::logic_error("[DataPacket_Hydro_gpu_3] CopyOutPerTile padding failure");

	// Copy out section
	nCopyToGpuBytes_ = nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded + nCopyInDataPerTileBytesPadded + nCopyInOutDataPerTileBytes;
	nReturnToHostBytes_ = nCopyInOutDataPerTileBytesPadded + nCopyOutDataPerTileBytes;
	std::size_t nBytesPerPacket = nScratchPerTileBytesPadded + nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded + nCopyInDataPerTileBytesPadded + nCopyInOutDataPerTileBytesPadded + nCopyOutDataPerTileBytesPadded;
	RuntimeBackend::instance().requestGpuMemory(nBytesPerPacket - nScratchPerTileBytesPadded, &packet_p_, nBytesPerPacket, &packet_d_);
	/// END

	/// POINTER DETERMINATION
	static_assert(sizeof(char) == 1);
	char* ptr_d = static_cast<char*>(packet_d_);

	// scratch section
	CC2_start_d_ = static_cast<void*>(ptr_d);
	ptr_d += nTiles * CC2_BLOCK_SIZE_HELPER;
	FCX_start_d_ = static_cast<void*>(ptr_d);
	ptr_d += nTiles * FCX_BLOCK_SIZE_HELPER;
	FCY_start_d_ = static_cast<void*>(ptr_d);
	ptr_d += nTiles * FCY_BLOCK_SIZE_HELPER;
	// end scratch

	location_ = PacketDataLocation::CC1;
	copyInStart_p_ = static_cast<char*>(packet_p_);
	copyInStart_d_ = static_cast<char*>(packet_d_) + nScratchPerTileBytesPadded;
	char* ptr_p = copyInStart_p_;
	ptr_d = copyInStart_d_;

	// general section;
	dt_start_p_ = static_cast<void*>(ptr_p);
	dt_start_d_ = static_cast<void*>(ptr_d);
	ptr_p += sizeof(dt_BLOCK_SIZE_HELPER);
	ptr_d += sizeof(dt_BLOCK_SIZE_HELPER);

	nTiles_start_p_ = static_cast<void*>(ptr_p);
	nTiles_start_d_ = static_cast<void*>(ptr_d);
	ptr_p += sizeof(nTiles_BLOCK_SIZE_HELPER);
	ptr_d += sizeof(nTiles_BLOCK_SIZE_HELPER);

	contents_p_ = static_cast<PacketContents*>( static_cast<void*>(ptr_p) );
	contents_d_ = static_cast<PacketContents*>( static_cast<void*>(ptr_d) );
	ptr_p += nTiles * sizeof(PacketContents);
	ptr_d += nTiles * sizeof(PacketContents);
	// end general

	// metadata section;
	deltas_start_p_ = static_cast<void*>(ptr_p);
	deltas_start_d_ = static_cast<void*>(ptr_d);
	ptr_p += nTiles * deltas_BLOCK_SIZE_HELPER;
	ptr_d += nTiles * deltas_BLOCK_SIZE_HELPER;

	lo_start_p_ = static_cast<void*>(ptr_p);
	lo_start_d_ = static_cast<void*>(ptr_d);
	ptr_p += nTiles * lo_BLOCK_SIZE_HELPER;
	ptr_d += nTiles * lo_BLOCK_SIZE_HELPER;

	hi_start_p_ = static_cast<void*>(ptr_p);
	hi_start_d_ = static_cast<void*>(ptr_d);
	ptr_p += nTiles * hi_BLOCK_SIZE_HELPER;
	ptr_d += nTiles * hi_BLOCK_SIZE_HELPER;

	// end metadata;

	// copy in section;
	ptr_p = copyInStart_p_ + nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded;
	ptr_d = copyInStart_d_ + nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded;

	// end copy in;

	// copy in out section
	copyInOutStart_p_ = copyInStart_p_ + nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded + nCopyInDataPerTileBytesPadded;
	copyInOutStart_d_ = copyInStart_d_ + nCopyInBytesPadded + nBlockMetadataPerTileBytesPadded + nCopyInDataPerTileBytesPadded;
	ptr_p = copyInOutStart_p_;
	ptr_d = copyInOutStart_d_;

	CC1_start_p_ = static_cast<void*>(ptr_p);
	CC1_start_d_ = static_cast<void*>(ptr_d);
	ptr_p += nTiles * CC1_BLOCK_SIZE_HELPER;
	ptr_d += nTiles * CC1_BLOCK_SIZE_HELPER;

	// end copy in out

	// copy out section
	// end copy out

	if (pinnedPtrs_) throw std::logic_error("DataPacket_Hydro_gpu_3::pack Pinned pointers already exist");
	pinnedPtrs_ = new BlockPointersPinned[nTiles];
	PacketContents* tilePtrs_p = contents_p_;
	char* char_ptr;
	/// END

	/// MEM COPY SECTION
	std::memcpy(dt_start_p_, static_cast<const void*>(&dt), dt_BLOCK_SIZE_HELPER);
	std::memcpy(nTiles_start_p_, static_cast<void*>(&nTiles), nTiles_BLOCK_SIZE_HELPER);

	for (std::size_t n=0; n < nTiles; ++n, ++tilePtrs_p) {
		Tile* tileDesc_h = tiles_[n].get();
		if (tileDesc_h == nullptr) throw std::runtime_error("[DataPacket_Hydro_gpu_3::pack] Bad tileDesc.");
		const RealVect deltas = tileDesc_h->deltas();
		const IntVect lo = tileDesc_h->lo();
		const IntVect hi = tileDesc_h->hi();
		Real* data_h = tileDesc_h->dataPtr();
		if (data_h == nullptr) throw std::logic_error("[DataPacket_Hydro_gpu_3::pack] Invalid ptr to data in host memory.");

		char_ptr = static_cast<char*>(deltas_start_p_) + n * deltas_BLOCK_SIZE_HELPER;
		Real deltas_h[MILHOJA_NDIM] = {deltas.I(), deltas.J(), deltas.K()}
		std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(deltas_h), deltas_BLOCK_SIZE_HELPER);

		char_ptr = static_cast<char*>(lo_start_p_) + n * lo_BLOCK_SIZE_HELPER;
		int lo_h[MILHOJA_NDIM] = {lo.I() + 1, lo.J() + 1, lo.K() + 1}
		std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(lo_h), lo_BLOCK_SIZE_HELPER);

		char_ptr = static_cast<char*>(hi_start_p_) + n * hi_BLOCK_SIZE_HELPER;
		int hi_h[MILHOJA_NDIM] = {hi.I() + 1, hi.J() + 1, hi.K() + 1}
		std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(hi_h), hi_BLOCK_SIZE_HELPER);

		std::size_t offset_CC1 = ((CC1_BLOCK_SIZE_HELPER / ( ((GAMC_VAR+UNK_VARS_BEGIN+1)) * sizeof(Real)) )) * static_cast<std::size_t>(UNK_VARS_BEGIN);
		std::size_t nBytes_CC1 = (GAMC_VAR - UNK_VARS_BEGIN + 1) * ((CC1_BLOCK_SIZE_HELPER / ( ((GAMC_VAR+UNK_VARS_BEGIN+1)) * sizeof(Real)) )) * sizeof(Real);
		char_ptr = static_cast<char*>(CC1_start_p_) + n * CC1_BLOCK_SIZE_HELPER;
		std::memcpy(static_cast<void*>(char_ptr), static_cast<const void*>(data_h + offset_CC1), nBytes_CC1);
		pinnedPtrs_[n].CC1_data = static_cast<Real*>( static_cast<void*>(char_ptr) );

	}
	/// END

	stream_ = RuntimeBackend::instance().requestStream(true);
	if (!stream_.isValid()) throw std::runtime_error("[DataPacket_Hydro_gpu_3::pack] Unable to acquire stream");
}

void DataPacket_Hydro_gpu_3::unpack(void) {
	using namespace milhoja;
	if (tiles_.size() <= 0) throw std::logic_error("[DataPacket_Hydro_gpu_3::unpack] Empty data packet.");
	if (!stream_.isValid()) throw std::logic_error("[DataPacket_Hydro_gpu_3::unpack] Stream not acquired.");
	if (pinnedPtrs_ == nullptr) throw std::logic_error("[DataPacket_Hydro_gpu_3::unpack] No pinned pointers set.");
	RuntimeBackend::instance().releaseStream(stream_);
	assert(!stream_.isValid());

	for (int n=0; n < tiles_.size(); ++n) {
		Tile* tileDesc_h = tiles_[n].get();
		Real* data_h = tileDesc_h->dataPtr();
		const Real* data_p = pinnedPtrs_[n].CC1_data;
		if (data_h == nullptr) throw std::logic_error("[DataPacket_Hydro_gpu_3::unpack] Invalid pointer to data in host memory.");
		if (data_p == nullptr) throw std::runtime_error("[DataPacket_Hydro_gpu_3::unpack] Invalid pointer to data in pinned memory.");
		assert(UNK_VARS_BEGIN == 0);
		assert(UNK_VARS_END == NUNKVAR - 1);

		std::size_t nBytes;
		if ( UNK_VARS_BEGIN < UNK_VARS_BEGIN || EINT_VAR < UNK_VARS_BEGIN || EINT_VAR > UNK_VARS_END || EINT_VAR - UNK_VARS_BEGIN + 1 > (GAMC_VAR+UNK_VARS_BEGIN+1))
				throw std::logic_error("[DataPacket_Hydro_gpu_3::unpack] Invalid variable mask");

		std::size_t offset_CC1 = ((CC1_BLOCK_SIZE_HELPER / ( ((GAMC_VAR+UNK_VARS_BEGIN+1)) * sizeof(Real)) )) * static_cast<std::size_t>(UNK_VARS_BEGIN);
		Real* start_h = data_h + offset_CC1;
		const Real* start_p_CC1 = data_p + offset_CC1;
		nBytes = (EINT_VAR - UNK_VARS_BEGIN + 1) * ((CC1_BLOCK_SIZE_HELPER / ( ((GAMC_VAR+UNK_VARS_BEGIN+1)) * sizeof(Real)) )) * sizeof(Real);
		std::memcpy(static_cast<void*>(start_h), static_cast<const void*>(start_p_CC1), nBytes);

	}
}

