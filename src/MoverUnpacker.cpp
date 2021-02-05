#include <iostream>
#include <stdexcept>

#include "MoverUnpacker.h"

#include "DataPacket.h"

namespace orchestration {

/**
 * Instantiate a MoverUnpacker extended finite state machine in the Idle state
 * with no data items in transit.
 */
MoverUnpacker::MoverUnpacker(void)
    : state_{State::Idle},
      nInTransit_{0} {
    pthread_cond_init(&unblockWaitThreads_, NULL);
    pthread_mutex_init(&mutex_, NULL);
}

/**
 * Destroy the eFSM.
 */
MoverUnpacker::~MoverUnpacker(void) {
    pthread_mutex_lock(&mutex_);

    if (nInTransit_ != 0) {
        std::cerr << "[MoverUnpacker::~MoverUnpacker] "
                  << "Data packets still in transit\n";
    } else if (state_ != State::Idle) {
        std::cerr << "[MoverUnpacker::~MoverUnpacker] "
                  << "MoverUnpacker is not Idle\n";
    }

    pthread_mutex_unlock(&mutex_);
    pthread_mutex_destroy(&mutex_);
    pthread_cond_destroy(&unblockWaitThreads_);
}

/**
 * Indicate to the eFSM that it should start accepting enqueued data items.  It
 * is a logical error for this to be called if the eFSM is not in Idle.
 */
void MoverUnpacker::startCycle(void) {
    pthread_mutex_lock(&mutex_);

    if (state_ != State::Idle) {
        pthread_mutex_unlock(&mutex_);
        throw std::logic_error("[MoverUnpacker::startCycle] "
                               "startCycle can only be called from Idle");
    } else if (nInTransit_ != 0) {
        pthread_mutex_unlock(&mutex_);
        throw std::runtime_error("[MoverUnpacker::startCycle] "
                                 "Number of items in transit not zero");
    }

    state_ = State::Open;

    pthread_mutex_unlock(&mutex_);
}

/**
 * The RuntimeElement interface requires defining this event.  However, it is a
 * logical error to call this as this eFSM cannot be a thread subscriber.
 */
void MoverUnpacker::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[MoverUnpacker::increaseThreadCount] "
                           "MoverUnpackers do no have threads to awaken");
}

/**
 * Use the calling thread to initiate the asynchronous transfer of the given
 * data packet from device to host.  This function also registers the
 * handleTransferFinished static function with the DataPacket.  See the
 * documentation for this routine for more information.
 *
 * \param  dataItem   The data packet that should be moved back to the host and
 *                    whose constituent data items should be enqueued with this
 *                    helper's data receiver if it has one.
 *
 * \todo This assumes that the calling thread need not be involved in the
 * unpacking process, which might not work for all runtime backends.  If it
 * does, can we get away with using preprocessors to accommodate the fact that
 * not all runtime backends will have identical interfaces to the
 * initiateDeviceToHostTransfer method?
 */
void MoverUnpacker::enqueue(std::shared_ptr<DataItem>&& dataItem) {
    pthread_mutex_lock(&mutex_);

    if (state_ != State::Open) {
        pthread_mutex_unlock(&mutex_);
        throw std::logic_error("[MoverUnpacker::enqueue] "
                               "enqueue only when in Open state");
    }
    ++nInTransit_;

    pthread_mutex_unlock(&mutex_);

    // The action taken here only makes sense if the given data item is a packet.
    // Therefore, this ugly upcasting is reasonable.
    DataPacket*    packet = dynamic_cast<DataPacket*>(dataItem.get());

    // Ownership of this memory resource will flow through the component
    // handling the transfer and back to the callback function, which must
    // release the resource.
    CallbackData*   userData = new CallbackData{};
    userData->unpacker = this;

    // Take ownership in such a way that we can transfer the ownership from
    // MoverUnpacker -> transfer component -> MoverUnpacker.
    // 
    // We need to continue wrapping the DataItem in a shared_ptr so that
    // multiple copies of this DataPacket can be flowing down different
    // pipelines simultaneously and still have correct release of resources when
    // the final shared_ptr exits its pipeline.
    // 
    // We need a dynamically-allocated shared_ptr, which is ugly in that it runs
    // counter to the intent of using smart pointers
    //    (i.e. don't use raw C++ pointers and
    //          limit use of new calls outside of constructors),
    // so that the shared_ptr persists as it moves to the data packet and back.
    userData->dataItem = new std::shared_ptr<DataItem>{ std::move(dataItem) };
    if ((dataItem.use_count() != 0) || (dataItem != nullptr)) {
        throw std::runtime_error("[MoverUnpacker::enqueue] "
                                 "shared_ptr not nullified as expected");
    }

    // This registers the callback and transfers ownership.
    packet->initiateDeviceToHostTransfer(handleTransferFinished, userData);
    userData = nullptr;
}

/**
 * One input/event to the eFSM is the transferFinished event that indicates when
 * an asynchronous transfer of a specific data packet has finished.  This event
 * is emitted by a data packet once it has arrived at the host and is emitted
 * by calling this function.  Therefore, It is intended that this static function
 * implement on behalf of the eFSM the appropriate triggering of outputs and
 * state transition associated with the occurrence of this event.  As this can
 * only be accomplished with knowledge of the actual state of the eFSM, the data
 * provided to this static function includes the MoverUnpacker object that is
 * managing the data packet and some responsibilities are carried out by the
 * object's handleTransferFinished_Stateful function.
 *
 * In particular, it is intended that this static function only be used as a
 * callback registered with a data packet when its transfer to host is
 * initiated.  This implies that this routine will be called by a host thread
 * outside of the thread team configuration when the packet's transfer is
 * complete.  This callback is, therefore, responsible for unpacking the
 * packet's data and enqueueing constituent data items with the data receiver of
 * the helper that handled the packet.  In addition, this function must close
 * the queue of the data receiver if it is working on the final data packet to
 * be transferred in the current runtime execution cycle.
 *
 * This routine not only assumes ownership of the given userData structure, and
 * therefore has the responsibility to release its resources, but also the
 * ownership of the contained DataPacket.  This routine will destroy this latter
 * shared_ptr so that the resources of its contents can also be released if
 * the given shared_ptr is the last copy of its kind.
 *
 * \param   userData - a pointer to an object of type CallbackData that contains
 *                     all data necessary for this static callback to carry out
 *                     its work.  This includes a pointer to the object that
 *                     registered the callback as well as the data item that was
 *                     transferred.
 */
void MoverUnpacker::handleTransferFinished(void* userData) {
    CallbackData*   data = static_cast<CallbackData*>(userData);
    if (!data) {
        throw std::logic_error("[MoverUnpacker::handleTransferFinished] Given null pointer");
    }
    MoverUnpacker*   unpacker = data->unpacker;
    if (!unpacker) {
        throw std::logic_error("[MoverUnpacker::handleTransferFinished] Given null unpacker");
    }
    RuntimeElement*  dataReceiver = unpacker->dataReceiver();
 
    // Unpacking only makes sense if the given data item is a packet.
    // Therefore, this ugly upcasting is reasonable.
    DataPacket*      packet = dynamic_cast<DataPacket*>(data->dataItem->get());
    if (!packet) {
        throw std::logic_error("[MoverUnpacker::handleTransferFinished] Given null packet");
    }
    packet->unpack();

    // Transfer the ownership of the data items in the packet to the next team
    if (dataReceiver) {
        while (packet->nTiles() > 0) {
            dataReceiver->enqueue( std::move(packet->popTile()) );
        }
        dataReceiver = nullptr;
    }
    packet = nullptr;

    // Allow for possibility that resources can be freed as soon as possible.
    // This shared_ptr to the DataPacket is no longer needed here or downstream
    // in the pipeline.
    delete data->dataItem;   data->dataItem = nullptr;
    data->unpacker = nullptr;
    delete data;   data = nullptr;

    unpacker->handleTransferFinished_Stateful();
    unpacker = nullptr;
}

/**
 * It is intended that this routine only be called by the callback function
 * handleTransferFinished once it has finished its work.  This is necessary as
 * some of the outputs required for transitions associated with the
 * transferFinished event must be based on the state of the MoverUnpacker
 * managing the movement of the data packet that issued the event.
 */
void  MoverUnpacker::handleTransferFinished_Stateful(void) {
    pthread_mutex_lock(&mutex_);

    if (nInTransit_ <= 0) {
        pthread_mutex_unlock(&mutex_);
        throw std::logic_error("[MoverUnpacker::handleTransferFinished_Stateful] "
                               "Callback count underflow");
    }
    --nInTransit_;

    if ((nInTransit_ == 0) && (state_ == State::Closed)) {
        if (dataReceiver_) {
            dataReceiver_->closeQueue();
        }
        state_ = State::Idle;
        pthread_cond_broadcast(&unblockWaitThreads_);
    }

    pthread_mutex_unlock(&mutex_);
}

/**
 * While this helper is presently implemented without a queue, we can
 * conceptually understand that the helper's "queue" is closed when this member
 * function is called in the sense that no data publishers should subsequently
 * call the helper's enqueue member function.  However, the queue of any data
 * subscriber will not be closed until the last data packet to be transferred in
 * the present runtime execution cycle by this helper has been handled by a
 * callback function.
 */
void MoverUnpacker::closeQueue(void) {
    pthread_mutex_lock(&mutex_);

    if (state_ != State::Open) {
        pthread_mutex_unlock(&mutex_);
        throw std::logic_error("[MoverUnpacker::closeQueue] "
                               "Queue can be closed only in Open state");
    }

    if (nInTransit_ == 0) {
        state_ = State::Idle;
        if (dataReceiver_) {
            dataReceiver_->closeQueue();
        }
        pthread_cond_broadcast(&unblockWaitThreads_);
    } else {
        state_ = State::Closed;
    }

    pthread_mutex_unlock(&mutex_);
}

/**
 * Block the calling thread until this object has finished transferring and
 * working on all data packets that will be enqueued with the object in the
 * current runtime execution cycle.  This function is non-blocking if the
 * current runtime execution cycle finished before the calling thread managed to
 * call this function.
 *
 * More than one thread can call wait during any given runtime execution cycle.
 */
void MoverUnpacker::wait(void) {
    pthread_mutex_lock(&mutex_);

    if (state_ != State::Idle) {
        pthread_cond_wait(&unblockWaitThreads_, &mutex_);
    }

    pthread_mutex_unlock(&mutex_);
}

}

