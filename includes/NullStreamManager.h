/**
 * \file    NullStreamManager.h
 *
 * \brief  Write this
 */

#ifndef NULL_STREAM_MANAGER_H__
#define NULL_STREAM_MANAGER_H__

#include <stdexcept>

#include "StreamManager.h"

#include "OrchestrationLogger.h"

namespace orchestration {

class NullStreamManager : public StreamManager {
public:
    ~NullStreamManager()
        { Logger::instance().log("[NullStreamManager] Finalized"); };

    int      numberFreeStreams(void) override
        { throw std::logic_error("[NullStreamManager::numberFreeStreams] Streams not available"); };

    Stream   requestStream(const bool block) override
        { throw std::logic_error("[NullStreamManager::requestStream] Streams not available"); };

    void     releaseStream(Stream& stream) override
        { throw std::logic_error("[NullStreamManager::releaseStream] Streams not available"); };

private:
    NullStreamManager()
        { Logger::instance().log("[NullStreamManager] Created and ready to not be used"); };

    // Needed for polymorphic singleton
    friend StreamManager& StreamManager::instance();
};

}

#endif

