#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void hip_omp_transform(double* values, int nValues, int* ranOnDevice);
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

double inputValue(const std::size_t i) {
    return 0.25 * static_cast<double>(i) - 3.0;
}

double expectedValue(const std::size_t i) {
    return 2.0 * inputValue(i) + 1.0;
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

        constexpr int nValues = 4096;
        const std::size_t nBytes =
            static_cast<std::size_t>(nValues) * sizeof(double);

        std::vector<double> hostValues(static_cast<std::size_t>(nValues));
        for (std::size_t i = 0; i < hostValues.size(); ++i) {
            hostValues[i] = inputValue(i);
        }

        DeviceBuffer deviceValues{nBytes};
        checkHip(hipMemcpy(deviceValues.get(), hostValues.data(), nBytes,
                           hipMemcpyHostToDevice),
                 "hipMemcpy host to device");

        int ranOnDevice = 0;
        hip_omp_transform(deviceValues.get(), nValues, &ranOnDevice);

        // The OpenMP target region is synchronous, but retain an explicit HIP
        // synchronization to make the ownership boundary under test obvious.
        checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
        checkHip(hipMemcpy(hostValues.data(), deviceValues.get(), nBytes,
                           hipMemcpyDeviceToHost),
                 "hipMemcpy device to host");

        if (ranOnDevice != 1) {
            throw std::runtime_error(
                "Fortran target region did not report device execution");
        }

        constexpr double tolerance = 1.0e-12;
        for (std::size_t i = 0; i < hostValues.size(); ++i) {
            const double expected = expectedValue(i);
            if (std::abs(hostValues[i] - expected) > tolerance) {
                throw std::runtime_error(
                    "Incorrect value at index " + std::to_string(i) +
                    ": got " + std::to_string(hostValues[i]) +
                    ", expected " + std::to_string(expected));
            }
        }

        std::cout << "HIP device: " << properties.name << '\n'
                  << "HIP allocation -> GNU Fortran OpenMP target -> HIP "
                     "copy-back passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HIP/OpenMP interoperability test failed: "
                  << error.what() << '\n';
        return 1;
    }
}

