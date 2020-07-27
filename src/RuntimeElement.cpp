#include "RuntimeElement.h"

#include <iostream>

using namespace orchestration;

RuntimeElement::RuntimeElement(void)
    : threadReceiver_{nullptr},
      dataReceiver_{nullptr}
{ }

RuntimeElement::~RuntimeElement(void) {
    if (threadReceiver_) {
        std::cerr << "[RuntimeElement::~RuntimeElement] Thread Subscriber still attached\n";
    }
    if (dataReceiver_) {
        std::cerr << "[RuntimeElement::~RuntimeElement] Data Subscriber still attached\n";
    }
}

/**
 * Attach given object as a thread subscriber.  Therefore, this converts
 * the calling object into a thread publisher.  A RuntimeElement shall be able
 * to attach to any other RuntimeElement as a subscriber regardless of the data
 * types of both teams. 
 *
 * It is a logic error
 *   - to attach an object to itself
 *   - to attach when an object is already attached.
 *
 * \param  receiver - the object of the desired thread subscriber
 */
std::string RuntimeElement::attachThreadReceiver(RuntimeElement* receiver) {
    // TODO: Do these need to be thread safe for classes like data item
    // splitters?
    if (!receiver) {
        return "Null thread subscriber given";
    } else if (receiver == this) {
        return "Cannot publish threads to itself";
    } else if (threadReceiver_) {
        return "Subcriber already attached";
    }

    threadReceiver_ = receiver;

    return "";
}

/**
 * Detach the thread subscriber so that the calling object is no longer a thread
 * publisher.
 */
std::string RuntimeElement::detachThreadReceiver(void) {
    if (!threadReceiver_) {
        return "No thread subscriber attached";
    }

    threadReceiver_ = nullptr;

    return "";
}

/**
 * Register given object as a data subscriber.  Therefore, this converts
 * the calling object into a data publisher.  A data publisher and data
 * subscriber must have the same same data type.
 *
 * \param  receiver - the team to which data items shall be published.
 */
std::string RuntimeElement::attachDataReceiver(RuntimeElement* receiver) {
    if (!receiver) {
        return "Null data subscriber given";
    } else if (receiver == this) {
        return "Cannot attach object to itself";
    } else if (dataReceiver_) {
        return "A data subscriber is already attached";
    }
    // TODO: Conform that publisher and subscriber are assigned the same data type?

    dataReceiver_ = receiver;

    return "";
}

/**
 * Detach the data subscriber so that the calling object is no longer a data
 * publisher.
 */
std::string RuntimeElement::detachDataReceiver(void) {
    if (!dataReceiver_) {
        return "No data subscriber attached";
    }

    dataReceiver_ = nullptr;
    
    return "";
}

