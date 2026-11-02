#include <cmath>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mpi.h>

#include <Milhoja.h>
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
#ifdef MILHOJA_CUDA_RUNTIME_BACKEND
constexpr const char* EXECUTION_NAME = "CUDA/OpenACC";
#else
constexpr const char* EXECUTION_NAME = "HOSTMEM";
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

milhoja::Real expectedChecksum(const int tile) {
    milhoja::Real checksum = 0.0;
    for (int variable = 0; variable < NVARS; ++variable) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                checksum += expectedValue(tile, i, j, variable);
            }
        }
    }
    return checksum;
}

milhoja::Real expectedReturnedValue(
    const int tile, const int i, const int j, const int variable
) {
    if (i == 0 && j == 0 && variable == 0) {
        return expectedChecksum(tile);
    }
    return expectedValue(tile, i, j, variable);
}

struct CallbackContext {
    DataPacket_roundtrip_tf* packet;
    std::mutex mutex;
    std::condition_variable ready;
    bool completed = false;
};

void unpackPacket(void* data) {
    auto* context = static_cast<CallbackContext*>(data);
    context->packet->unpack();
    {
        std::lock_guard<std::mutex> lock(context->mutex);
        context->completed = true;
    }
    context->ready.notify_one();
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

#ifdef MILHOJA_HOSTMEM_RUNTIME_BACKEND
            for (int tile = 0; tile < NTILES; ++tile) {
                const milhoja::Real expected = expectedChecksum(tile);
                const milhoja::Real actual =
                    packet._scratch_checksum_d[tile];
                require(
                    std::abs(actual - expected) <= 1.0e-12 *
                        (1.0 + std::abs(expected)),
                    "scratch checksum mismatch for tile " +
                        std::to_string(tile)
                );
            }
#endif

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

            CallbackContext callback;
            callback.packet = &packet;
            backend.initiateGpuToHostTransfer(
                packet, unpackPacket, static_cast<void*>(&callback)
            );
            {
                std::unique_lock<std::mutex> lock(callback.mutex);
                callback.ready.wait(lock, [&callback]() {
                    return callback.completed;
                });
            }
            require(
                backend.numberFreeStreams() == 1,
                "unpack did not release the runtime stream"
            );

            for (int tile = 0; tile < NTILES; ++tile) {
                for (int variable = 0; variable < NVARS; ++variable) {
                    for (int j = 0; j < NY; ++j) {
                        for (int i = 0; i < NX; ++i) {
                            const milhoja::Real actual =
                                fields[tile][offset(i, j, 0, variable)];
                            const milhoja::Real expected =
                                expectedReturnedValue(
                                    tile, i, j, variable
                                );
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
        std::cout << "Milhoja interoperability " << EXECUTION_NAME << " "
                  << LAYOUT_NAME
                  << " round trip passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
