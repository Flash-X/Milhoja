#include "Milhoja_RuntimeElement.h"

#include <iostream>

using namespace milhoja;

RuntimeElement::RuntimeElement(void)
    : threadReceiver_{nullptr},
      dataReceiver_{nullptr},
      receiverPrototype_{nullptr},
      calledCloseQueue_{}
{ }

RuntimeElement::~RuntimeElement(void) {
    if (threadReceiver_) {
        std::cerr << "[RuntimeElement::~RuntimeElement] Thread Subscriber still attached\n";
    }
    if (dataReceiver_) {
        std::cerr << "[RuntimeElement::~RuntimeElement] Data Subscriber still attached\n";
    }
    if (receiverPrototype_) {
        std::cerr << "[RuntimeElement::~RuntimeElement] Receiver Prototype still set\n";
    }
    if (!calledCloseQueue_.empty()) {
        std::cerr << "[RuntimeElement::~RuntimeElement] Data publishers still attached\n";
        // FIXME: Does this help prevent valgrind from finding potential pointer
        // issues? 
        while (!calledCloseQueue_.empty()) {
            calledCloseQueue_.erase(calledCloseQueue_.begin());
        }
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
 * the calling object into a data publisher.
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

    // Establish the two-way communication
    dataReceiver_ = receiver;
    return dataReceiver_->attachDataPublisher(this);
}

/**
 * Detach the data subscriber so that the calling object is no longer a data
 * publisher.
 */
std::string RuntimeElement::detachDataReceiver(void) {
    if (!dataReceiver_) {
        return "No data subscriber attached";
    }

    // Completely breakdown two-way communication
    std::string  errMsg = dataReceiver_->detachDataPublisher(this);
    if (errMsg != "") {
        return errMsg;
    }

    dataReceiver_ = nullptr;

    // if it has a receiver's prototype, release it
    receiverPrototype_ = nullptr;

    return "";
}

/**
 * This member function should only be called by attachDataReceiver as part of
 * establishing the two-way communication between the data publisher and
 * subscriber.
 *
 * \param  publisher - the RuntimeElement that is registering itself with the
 *                     object as one of its (possibly many) data publishers.
 *                     A null pointer is not a valid publisher.
 *
 * \return An empty string if successful.  Otherwise, an error message.
 */
std::string RuntimeElement::attachDataPublisher(const RuntimeElement* publisher) {
    if (!publisher) {
        return "Given publisher is null";
    } else if (calledCloseQueue_.find(publisher) != calledCloseQueue_.end()) {
        return "Given publisher already attached as a publisher";
    }

    calledCloseQueue_[publisher] = false;

    return "";
}

/**
 * This member function should only be called by detachDataReceiver as part of
 * breaking down the two-way communication between the data publisher and
 * subscriber.
 *
 * \param  publisher - the RuntimeElement that was registered with the object
 *                     as one of its (possibly many) data publishers and that
 *                     will no longer publish to the object.
 *
 * \return An empty string if successful.  Otherwise, an error message.
 */
std::string RuntimeElement::detachDataPublisher(const RuntimeElement* publisher) {
    std::map<const RuntimeElement*,bool>::iterator    itor = calledCloseQueue_.find(publisher);
    if (itor == calledCloseQueue_.end()) {
        return "Given publisher never attached as a publisher";
    }

    // TODO: Should this fail if the key has an unacceptable value?
    calledCloseQueue_.erase(itor);

    return "";
}

/**
 * Set the data receiver's prototype for later use when passing
 * a DataItem to the data receiver, for calling a proper constructor.
 * Note that the receiver's prototype is only required for passing TilwWrapper, currently.
 * Thus, calling this function for the DataPacket has no effect.
 * The receiverPrototype_ will be nullified when RuntimeElement::detachDataReceiver is called.
 *
 * \param prototype - A prototype of a DataItem to be passed to the DataReceiver.
 */
std::string RuntimeElement::setReceiverPrototype(const DataItem* prototype) {

    if (!prototype) {
        return "Null receiver prototype is given";
    } else if (receiverPrototype_) {
        return "A receiver prototype is already given";
    }

    receiverPrototype_ = prototype;

    return "";
}

