#include "Milhoja_HipStreamManager.h"

#include <climits>
#include <iostream>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

namespace milhoja {

bool HipStreamManager::initialized_ = false;
bool HipStreamManager::finalized_ = false;
int HipStreamManager::nMaxStreams_ = -1;

void HipStreamManager::initialize(const int nMaxStreams) {
    if (initialized_) {
        throw std::logic_error("[HipStreamManager::initialize] Already initialized");
    }
    if (nMaxStreams <= 0) {
        throw std::invalid_argument("[HipStreamManager::initialize] Need a stream");
    }
    nMaxStreams_ = nMaxStreams;
    initialized_ = true;
    instance();
}

HipStreamManager& HipStreamManager::instance(void) {
    if (!initialized_) {
        throw std::logic_error("[HipStreamManager::instance] Not initialized");
    }
    if (finalized_) {
        throw std::logic_error("[HipStreamManager::instance] Already finalized");
    }
    static HipStreamManager manager;
    return manager;
}

HipStreamManager::HipStreamManager(void) : streams_{} {
    pthread_mutex_init(&mutex_, nullptr);
    pthread_cond_init(&streamReleased_, nullptr);
    for (int i = 0; i < nMaxStreams_; ++i) {
        Stream stream{};
        const hipError_t error = hipStreamCreate(&stream.hipStream);
        if (error != hipSuccess) {
            throw std::runtime_error(
                "[HipStreamManager] hipStreamCreate: " +
                std::string(hipGetErrorString(error))
            );
        }
        streams_.push_back(std::move(stream));
    }
}

void HipStreamManager::finalize(void) {
    if (!initialized_ || finalized_) {
        throw std::logic_error("[HipStreamManager::finalize] Invalid state");
    }
    pthread_mutex_lock(&mutex_);
    if (static_cast<int>(streams_.size()) != nMaxStreams_) {
        pthread_mutex_unlock(&mutex_);
        throw std::runtime_error("[HipStreamManager::finalize] Streams still acquired");
    }
    for (Stream& stream : streams_) {
        const hipError_t error = hipStreamDestroy(stream.hipStream);
        if (error != hipSuccess) {
            pthread_mutex_unlock(&mutex_);
            throw std::runtime_error(
                "[HipStreamManager] hipStreamDestroy: " +
                std::string(hipGetErrorString(error))
            );
        }
        stream.hipStream = nullptr;
    }
    streams_.clear();
    pthread_mutex_unlock(&mutex_);
    pthread_cond_destroy(&streamReleased_);
    pthread_mutex_destroy(&mutex_);
    nMaxStreams_ = -1;
    finalized_ = true;
}

int HipStreamManager::numberFreeStreams(void) {
    pthread_mutex_lock(&mutex_);
    const std::size_t count = streams_.size();
    pthread_mutex_unlock(&mutex_);
    if (count > static_cast<std::size_t>(INT_MAX)) {
        throw std::overflow_error("[HipStreamManager] Too many streams");
    }
    return static_cast<int>(count);
}

Stream HipStreamManager::requestStream(const bool block) {
    pthread_mutex_lock(&mutex_);
    while (streams_.empty()) {
        if (!block) {
            pthread_mutex_unlock(&mutex_);
            return Stream{};
        }
        pthread_cond_wait(&streamReleased_, &mutex_);
    }
    Stream stream = std::move(streams_.front());
    streams_.pop_front();
    pthread_mutex_unlock(&mutex_);
    return stream;
}

void HipStreamManager::releaseStream(Stream& stream) {
    if (!stream.isValid()) {
        throw std::invalid_argument("[HipStreamManager::releaseStream] Invalid stream");
    }
    pthread_mutex_lock(&mutex_);
    if (static_cast<int>(streams_.size()) >= nMaxStreams_) {
        pthread_mutex_unlock(&mutex_);
        throw std::invalid_argument("[HipStreamManager::releaseStream] All streams free");
    }
    streams_.push_back(std::move(stream));
    pthread_cond_signal(&streamReleased_);
    pthread_mutex_unlock(&mutex_);
}

HipStreamManager::~HipStreamManager(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[HipStreamManager::~HipStreamManager] ERROR - Not finalized\n";
    }
}

}
