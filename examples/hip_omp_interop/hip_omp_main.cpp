#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void hip_omp_transform_variable_last(double* values,
                                     int nX,
                                     int nY,
                                     int nZ,
                                     int nVariables,
                                     int* ranOnDevice);
void hip_omp_transform_variable_first(double* values,
                                      int nX,
                                      int nY,
                                      int nZ,
                                      int nVariables,
                                      int* ranOnDevice);
}

namespace {

void checkHip(const hipError_t code, const char* operation) {
    if (code != hipSuccess) {
        throw std::runtime_error(
            std::string{operation} + ": " + hipGetErrorString(code));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t nBytes) {
        checkHip(hipMalloc(&data_, nBytes), "hipMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            const hipError_t code = hipFree(data_);
            if (code != hipSuccess) {
                std::cerr << "hipFree during cleanup: "
                          << hipGetErrorString(code) << '\n';
            }
        }
    }

    double* get() noexcept {
        return static_cast<double*>(data_);
    }

private:
    void* data_{nullptr};
};

enum class Layout {
    VariableLast,
    VariableFirst
};

using Transform = void (*)(double*, int, int, int, int, int*);

constexpr int nX = 5;
constexpr int nY = 3;
constexpr int nZ = 2;
constexpr int nVariables = 4;

std::size_t variableLastOffset(const int i,
                               const int j,
                               const int k,
                               const int variable) {
    return static_cast<std::size_t>(i)
        + static_cast<std::size_t>(nX)
              * (static_cast<std::size_t>(j)
                 + static_cast<std::size_t>(nY)
                       * (static_cast<std::size_t>(k)
                          + static_cast<std::size_t>(nZ)
                                * static_cast<std::size_t>(variable)));
}

std::size_t variableFirstOffset(const int i,
                                const int j,
                                const int k,
                                const int variable) {
    return static_cast<std::size_t>(variable)
        + static_cast<std::size_t>(nVariables)
              * (static_cast<std::size_t>(i)
                 + static_cast<std::size_t>(nX)
                       * (static_cast<std::size_t>(j)
                          + static_cast<std::size_t>(nY)
                                * static_cast<std::size_t>(k)));
}

std::size_t physicalOffset(const Layout layout,
                           const int i,
                           const int j,
                           const int k,
                           const int variable) {
    return layout == Layout::VariableLast
               ? variableLastOffset(i, j, k, variable)
               : variableFirstOffset(i, j, k, variable);
}

double logicalIndex(const int i,
                    const int j,
                    const int k,
                    const int variable) {
    return static_cast<double>(i + 10 * j + 100 * k + 1000 * variable);
}

double inputValue(const int i,
                  const int j,
                  const int k,
                  const int variable) {
    return 0.125 * logicalIndex(i, j, k, variable) - 3.0;
}

double expectedValue(const int i,
                     const int j,
                     const int k,
                     const int variable) {
    return 1.5 * inputValue(i, j, k, variable)
        + logicalIndex(i, j, k, variable);
}

void runLayout(const Layout layout,
               const char* name,
               const Transform transform) {
    constexpr std::size_t nValues =
        static_cast<std::size_t>(nX) * nY * nZ * nVariables;
    constexpr std::size_t nBytes = nValues * sizeof(double);

    std::vector<double> hostValues(
        nValues, std::numeric_limits<double>::quiet_NaN());
    std::vector<bool> initialized(nValues, false);

    for (int variable = 0; variable < nVariables; ++variable) {
        for (int k = 0; k < nZ; ++k) {
            for (int j = 0; j < nY; ++j) {
                for (int i = 0; i < nX; ++i) {
                    const std::size_t offset =
                        physicalOffset(layout, i, j, k, variable);
                    if (offset >= nValues || initialized[offset]) {
                        throw std::runtime_error(
                            std::string{name} +
                            " host offset mapping is not one-to-one");
                    }
                    initialized[offset] = true;
                    hostValues[offset] = inputValue(i, j, k, variable);
                }
            }
        }
    }

    if (!std::all_of(initialized.begin(), initialized.end(),
                     [](const bool value) { return value; })) {
        throw std::runtime_error(
            std::string{name} + " host offset mapping has holes");
    }

    DeviceBuffer deviceValues{nBytes};
    checkHip(hipMemcpy(deviceValues.get(), hostValues.data(), nBytes,
                       hipMemcpyHostToDevice),
             "hipMemcpy host to device");

    int ranOnDevice = 0;
    transform(deviceValues.get(), nX, nY, nZ, nVariables, &ranOnDevice);

    // The OpenMP target construct is synchronous. Retain an explicit HIP
    // synchronization to make the cross-runtime ownership boundary obvious.
    checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    checkHip(hipMemcpy(hostValues.data(), deviceValues.get(), nBytes,
                       hipMemcpyDeviceToHost),
             "hipMemcpy device to host");

    if (ranOnDevice != 1) {
        throw std::runtime_error(
            std::string{name} +
            " Fortran region did not report device execution");
    }

    constexpr double tolerance = 1.0e-12;
    for (int variable = 0; variable < nVariables; ++variable) {
        for (int k = 0; k < nZ; ++k) {
            for (int j = 0; j < nY; ++j) {
                for (int i = 0; i < nX; ++i) {
                    const std::size_t offset =
                        physicalOffset(layout, i, j, k, variable);
                    const double expected = expectedValue(i, j, k, variable);
                    if (std::abs(hostValues[offset] - expected) > tolerance) {
                        throw std::runtime_error(
                            std::string{name} + " incorrect logical value (" +
                            std::to_string(i) + "," + std::to_string(j) +
                            "," + std::to_string(k) + "," +
                            std::to_string(variable) + "): got " +
                            std::to_string(hostValues[offset]) +
                            ", expected " + std::to_string(expected));
                    }
                }
            }
        }
    }

    std::cout << name << " layout passed\n";
}

} // namespace

int main() {
    try {
        int deviceCount = 0;
        checkHip(hipGetDeviceCount(&deviceCount), "hipGetDeviceCount");
        if (deviceCount < 1) {
            throw std::runtime_error("HIP reports no visible GPU");
        }

        checkHip(hipSetDevice(0), "hipSetDevice");

        hipDeviceProp_t properties{};
        checkHip(hipGetDeviceProperties(&properties, 0),
                 "hipGetDeviceProperties");

        runLayout(Layout::VariableLast, "variable-last",
                  hip_omp_transform_variable_last);
        runLayout(Layout::VariableFirst, "variable-first",
                  hip_omp_transform_variable_first);

        std::cout << "HIP device: " << properties.name << '\n'
                  << "HIP allocation -> GNU Fortran OpenMP target -> HIP "
                     "copy-back passed for both layouts\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HIP/OpenMP interoperability test failed: "
                  << error.what() << '\n';
        return 1;
    }
}

