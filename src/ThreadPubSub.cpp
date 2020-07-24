#include "ThreadPubSub.h"

#include <iostream>

using namespace orchestration;

ThreadPubSub::ThreadPubSub(void)
    : threadReceiver_{nullptr}
{ }

ThreadPubSub::~ThreadPubSub(void) {
    if (threadReceiver_) {
        std::cerr << "[ThreadPubSub::~ThreadPubSub] Subscriber still attached\n";
    }
}

/**
 * Attach given object as a thread subscriber.  Therefore, this converts
 * the calling object into a thread publisher.  A ThreadPubSub shall be able
 * to attach to any other thread team as a subscriber regardless of the data
 * types of both teams. 
 *
 * It is a logic error
 *   - to attach an object to itself
 *   - to attach when an object is already attached.
 *
 * \param  receiver - the object of the desired thread subscriber
 */
std::string ThreadPubSub::attachThreadReceiver(ThreadPubSub* receiver) {
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
std::string ThreadPubSub::detachThreadReceiver(void) {
    if (!threadReceiver_) {
        return "No thread subscriber attached";
    }

    threadReceiver_ = nullptr;

    return "";
}

