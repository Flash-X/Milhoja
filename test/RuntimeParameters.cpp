#include "RuntimeParameters.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <Milhoja_Logger.h>

std::string   RuntimeParameters::filename_    = "";
bool          RuntimeParameters::initialized_ = false;
bool          RuntimeParameters::finalized_   = false;

/**
 *
 */
void   RuntimeParameters::initialize(const std::string& filename) {
    // finalized_ => initialized_
    // Therefore no need to check finalized_
    if (initialized_) {
        throw std::logic_error("[RuntimeParameters::initialize] Already initialized");
    }

    milhoja::Logger::instance().log("[RuntimeParameters] Initializing...");

    filename_ = filename;
    initialized_ = true;
    instance();

    milhoja::Logger::instance().log("[RuntimeParameters] Initialized");
}

/**
 *
 */
RuntimeParameters&    RuntimeParameters::instance(void) {
    if        (!initialized_) {
        throw std::logic_error("[RuntimeParameters::instance] Not initialized");
    } else if (finalized_) {
        throw std::logic_error("[RuntimeParameters::instance] No access after finalization");
    }

    static RuntimeParameters   singleton;
    return singleton;
}

/**
 *
 */
void   RuntimeParameters::finalize(void) {
    if        (!initialized_) {
        throw std::logic_error("[RuntimeParameters::finalize] Never initialized");
    } else if (finalized_) {
        throw std::logic_error("[RuntimeParameters::finalize] Already finalized");
    }

    milhoja::Logger::instance().log("[RuntimeParameters] Finalizing...");

    finalized_ = true;

    milhoja::Logger::instance().log("[RuntimeParameters] Finalized");
}

/**
 *
 */
RuntimeParameters::RuntimeParameters(void) {
    std::ifstream   jsonStream(filename_);
    if (!jsonStream.good()) {
        throw std::invalid_argument("[RuntimeParameters::RuntimeParameters] Invalid file");
    }

    jsonStream >> json_;

    milhoja::Logger::instance().log("[RuntimeParameters] Loaded values from " + filename_);
}

/**
 *
 */
RuntimeParameters::~RuntimeParameters(void) {
    if (initialized_ && !finalized_) {
        std::cerr << "[RuntimeParamters::~RuntimeParameters] Not finalized"
                  << std::endl;
    }
}

/**
 *
 */
int    RuntimeParameters::getInt(const std::string& group,
                                 const std::string& parameter) const {
    if (finalized_) {
        throw std::logic_error("[RuntimeParameters::getInt] Invalid after finalize");
    }

    return json_.at(group).at(parameter).get<int>();
}

/**
 *
 */
unsigned int    RuntimeParameters::getUnsignedInt(const std::string& group,
                                                  const std::string& parameter) const {
    if (finalized_) {
        throw std::logic_error("[RuntimeParameters::getUnsignedInt] Invalid after finalize");
    }

    return json_[group][parameter].get<unsigned int>();
}

/**
 *
 */
std::size_t    RuntimeParameters::getSizeT(const std::string& group,
                                           const std::string& parameter) const {
    if (finalized_) {
        throw std::logic_error("[RuntimeParameters::getSizeT] Invalid after finalize");
    }

    return json_[group][parameter].get<std::size_t>();
}

/**
 *
 */
milhoja::Real   RuntimeParameters::getReal(const std::string& group,
                                           const std::string& parameter) const {
    if (finalized_) {
        throw std::logic_error("[RuntimeParameters::getReal] Invalid after finalize");
    }

    return json_[group][parameter].get<milhoja::Real>();
}

