#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>

#include <Milhoja_FlashxrTileRaw.h>
#include <Milhoja_Logger.h>
#include <Milhoja_RuntimeBackend.h>
#include <Milhoja_TileFlashxr.h>

#include "DataPacket_roundtrip_tf.h"

extern "C" void roundtrip_tf_Cpp2C(int threadIndex, void* dataItem);

namespace {

constexpr int NX = 5;
constexpr int NY = 4;
constexpr int NZ = 1;
constexpr int NVARS = 4;
constexpr int NTILES = 2;
constexpr milhoja::Real SCALE = 2.5;
#ifdef ROUNDTRIP_VARIABLE_FIRST
constexpr const char* LAYOUT_NAME = "variable-first";
#else
constexpr const char* LAYOUT_NAME = "variable-last";
#endif

std::size_t offset(const int i, const int j, const int k, const int variable) {
    return static_cast<std::size_t>(i) + NX * (
        static_cast<std::size_t>(j) + NY * (
            static_cast<std::size_t>(k) + NZ * variable));
}

milhoja::Real initialValue(
    const int tile, const int i, const int j, const int variable
) {
    return static_cast<milhoja::Real>(
        1000 * (tile + 1) + 100 * (variable + 1) + 10 * (j + 1) + (i + 1)
    );
}

milhoja::Real expectedValue(
    const int tile, const int i, const int j, const int variable
) {
    constexpr int metadataCode = 1;  // tile_lbound = [0, 0, 1]
    const milhoja::Real increment = static_cast<milhoja::Real>(
        100 * (variable + 1) + 10 * (j + 1) + (i + 1) + metadataCode
    );
    return SCALE * initialValue(tile, i, j, variable) + increment;
}

struct CallbackContext {
    DataPacket_roundtrip_tf* packet;
    bool completed;
};

void unpackPacket(void* data) {
    auto* context = static_cast<CallbackContext*>(data);
    context->packet->unpack();
    context->completed = true;
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    bool ownsMpi = false;
    try {
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if (!mpiInitialized) {
            MPI_Init(nullptr, nullptr);
            ownsMpi = true;
        }

        constexpr std::size_t poolBytes = 1024 * 1024;
        milhoja::RuntimeBackend::initialize(1, poolBytes, poolBytes);

        {
            std::vector<std::vector<milhoja::Real>> fields(
                NTILES, std::vector<milhoja::Real>(NX * NY * NZ * NVARS)
            );
            for (int tile = 0; tile < NTILES; ++tile) {
                for (int variable = 0; variable < NVARS; ++variable) {
                    for (int j = 0; j < NY; ++j) {
                        for (int i = 0; i < NX; ++i) {
                            fields[tile][offset(i, j, 0, variable)] =
                                initialValue(tile, i, j, variable);
                        }
                    }
                }
            }

            DataPacket_roundtrip_tf packet{SCALE};
            // DataPacket::addTile pushes to the front. Add in reverse so the
            // packet tile axis is ordered [tile 0, tile 1].
            for (int tile = NTILES - 1; tile >= 0; --tile) {
                FlashxrTileRawPtrs pointers{
                    fields[tile].data(), nullptr, nullptr, nullptr
                };
                FlashxTileRawInts integers{
                    NVARS, 0,
                    0, 0, 1,
                    4, 3, 1,
                    1, 1, 1,
                    3, 2, 1,
                    2, 1, tile + 1, 1
                };
                FlashxTileRawReals reals{1.0, 1.0, 1.0};
                std::shared_ptr<milhoja::Tile> descriptor =
                    std::make_shared<milhoja::TileFlashxr>(
                        pointers, integers, reals
                    );
                packet.addTile(std::move(descriptor));
            }

            packet.pack();
            milhoja::RuntimeBackend& backend =
                milhoja::RuntimeBackend::instance();
            require(
                backend.numberFreeStreams() == 0,
                "pack did not acquire the only HOSTMEM stream"
            );
            backend.initiateHostToGpuTransfer(packet);
            roundtrip_tf_Cpp2C(0, &packet);

            std::vector<milhoja::Real> expectedChecksums(NTILES, 0.0);
            for (int tile = 0; tile < NTILES; ++tile) {
                for (int variable = 0; variable < NVARS; ++variable) {
                    for (int j = 0; j < NY; ++j) {
                        for (int i = 0; i < NX; ++i) {
                            expectedChecksums[tile] +=
                                expectedValue(tile, i, j, variable);
                        }
                    }
                }
                const milhoja::Real actual =
                    packet._scratch_checksum_d[tile];
                require(
                    std::abs(actual - expectedChecksums[tile]) <= 1.0e-12 *
                        (1.0 + std::abs(expectedChecksums[tile])),
                    "scratch checksum mismatch for tile " +
                        std::to_string(tile)
                );
            }

            // Tile storage must not change until the return transfer invokes
            // unpack. HOSTMEM still uses distinct packet memory.
            for (int tile = 0; tile < NTILES; ++tile) {
                for (int variable = 0; variable < NVARS; ++variable) {
                    for (int j = 0; j < NY; ++j) {
                        for (int i = 0; i < NX; ++i) {
                            require(
                                fields[tile][offset(i, j, 0, variable)] ==
                                    initialValue(tile, i, j, variable),
                                "Tile storage changed before unpack"
                            );
                        }
                    }
                }
            }

            CallbackContext callback{&packet, false};
            backend.initiateGpuToHostTransfer(
                packet, unpackPacket, static_cast<void*>(&callback)
            );
            require(
                callback.completed,
                "HOSTMEM return callback did not complete synchronously"
            );
            require(
                backend.numberFreeStreams() == 1,
                "unpack did not release the HOSTMEM stream"
            );

            for (int tile = 0; tile < NTILES; ++tile) {
                for (int variable = 0; variable < NVARS; ++variable) {
                    for (int j = 0; j < NY; ++j) {
                        for (int i = 0; i < NX; ++i) {
                            const milhoja::Real actual =
                                fields[tile][offset(i, j, 0, variable)];
                            const milhoja::Real expected =
                                expectedValue(tile, i, j, variable);
                            require(
                                actual == expected,
                                "field mismatch at tile=" +
                                    std::to_string(tile) + " variable=" +
                                    std::to_string(variable) + " i=" +
                                    std::to_string(i) + " j=" +
                                    std::to_string(j)
                            );
                        }
                    }
                }
            }
        }

        milhoja::RuntimeBackend::instance().finalize();
        milhoja::Logger::instance().finalize();
        if (ownsMpi) {
            MPI_Finalize();
        }
        std::cout << "Milhoja Step 5D HOSTMEM " << LAYOUT_NAME
                  << " round trip passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
